#include "text_renderer.h"

// Single-header image loading library - define implementation here
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace TextRenderer {

    // Global state
    GLuint fontTexture = 0;
    GLuint textShaderProgram = 0;
    GLuint textVAO = 0, textVBO = 0;
    int screenWidth = 800;
    int screenHeight = 600;

    // Shader sources
    static const char* textVertexShaderSrc = R"(
        #version 430 core
        
        layout(location = 0) in vec4 vertex;  // <vec2 pos, vec2 tex>
        
        out vec2 TexCoords;
        
        uniform mat4 projection;
        
        void main() {
            gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
            TexCoords = vertex.zw;
        }
    )";

    static const char* textFragmentShaderSrc = R"(
        #version 430 core
        
        in vec2 TexCoords;
        out vec4 FragColor;
        
        uniform sampler2D fontTexture;
        uniform vec4 textColor;
        
        void main() {
            // Sample the font texture
            vec4 sampled = texture(fontTexture, TexCoords);
            
            // Use luminance as alpha (white text on transparent background)
            float alpha = max(sampled.r, max(sampled.g, sampled.b));
            
            // If the texture has an alpha channel, use it
            if (sampled.a < 1.0) {
                alpha = sampled.a;
            }
            
            // Discard nearly transparent fragments
            if (alpha < 0.1) {
                discard;
            }
            
            FragColor = vec4(textColor.rgb, textColor.a * alpha);
        }
    )";

    // Helper function to compile shaders
    static GLuint compileShader(GLenum type, const char* source) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(shader, 512, nullptr, infoLog);
            std::cerr << "TextRenderer: Shader compilation error: " << infoLog << std::endl;
            return 0;
        }
        return shader;
    }

    // Helper function to create shader program
    static GLuint createShaderProgram(const char* vertSrc, const char* fragSrc) {
        GLuint vertShader = compileShader(GL_VERTEX_SHADER, vertSrc);
        GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, fragSrc);

        if (!vertShader || !fragShader) {
            return 0;
        }

        GLuint program = glCreateProgram();
        glAttachShader(program, vertShader);
        glAttachShader(program, fragShader);
        glLinkProgram(program);

        GLint success;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(program, 512, nullptr, infoLog);
            std::cerr << "TextRenderer: Program linking error: " << infoLog << std::endl;
            return 0;
        }

        glDeleteShader(vertShader);
        glDeleteShader(fragShader);

        return program;
    }

    bool init(const char* fontPath) {
        // Load font texture
        int width, height, channels;
        stbi_set_flip_vertically_on_load(true);  // OpenGL expects bottom-to-top
        unsigned char* data = stbi_load(fontPath, &width, &height, &channels, 4);

        if (!data) {
            std::cerr << "TextRenderer: Failed to load font texture: " << fontPath << std::endl;
            return false;
        }

        std::cout << "TextRenderer: Loaded font " << width << "x" << height
            << " with " << channels << " channels" << std::endl;

        // Create OpenGL texture
        glGenTextures(1, &fontTexture);
        glBindTexture(GL_TEXTURE_2D, fontTexture);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);

        // Create shader program
        textShaderProgram = createShaderProgram(textVertexShaderSrc, textFragmentShaderSrc);
        if (!textShaderProgram) {
            std::cerr << "TextRenderer: Failed to create shader program" << std::endl;
            return false;
        }

        // Create VAO and VBO for text quads
        // We'll update VBO data dynamically for each character
        glGenVertexArrays(1, &textVAO);
        glGenBuffers(1, &textVBO);

        glBindVertexArray(textVAO);
        glBindBuffer(GL_ARRAY_BUFFER, textVBO);

        // Allocate buffer for a quad (6 vertices * 4 floats each)
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, nullptr, GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        std::cout << "TextRenderer: Initialized successfully" << std::endl;
        return true;
    }

    void cleanup() {
        if (fontTexture) {
            glDeleteTextures(1, &fontTexture);
            fontTexture = 0;
        }
        if (textShaderProgram) {
            glDeleteProgram(textShaderProgram);
            textShaderProgram = 0;
        }
        if (textVAO) {
            glDeleteVertexArrays(1, &textVAO);
            textVAO = 0;
        }
        if (textVBO) {
            glDeleteBuffers(1, &textVBO);
            textVBO = 0;
        }
    }

    void setScreenSize(int width, int height) {
        screenWidth = width;
        screenHeight = height;
    }

    float getTextWidth(const std::string& text, float scale) {
        // Each character is CHAR_PIXEL_SIZE pixels wide
        return text.length() * CHAR_PIXEL_SIZE * scale;
    }

    float getCharHeight(float scale) {
        return CHAR_PIXEL_SIZE * scale;
    }

    void drawText(float x, float y, const std::string& text, float scale, const glm::vec4& color) {
        if (!textShaderProgram || !fontTexture || text.empty()) {
            return;
        }

        // Save current state
        GLboolean depthTestEnabled;
        GLboolean blendEnabled;
        GLint blendSrc, blendDst;
        glGetBooleanv(GL_DEPTH_TEST, &depthTestEnabled);
        glGetBooleanv(GL_BLEND, &blendEnabled);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &blendSrc);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &blendDst);

        // Setup rendering state for 2D text
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Use text shader
        glUseProgram(textShaderProgram);

        // Set orthographic projection (screen space)
        glm::mat4 projection = glm::ortho(0.0f, (float)screenWidth, (float)screenHeight, 0.0f);
        glUniformMatrix4fv(glGetUniformLocation(textShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform4fv(glGetUniformLocation(textShaderProgram, "textColor"), 1, glm::value_ptr(color));

        // Bind font texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, fontTexture);
        glUniform1i(glGetUniformLocation(textShaderProgram, "fontTexture"), 0);

        glBindVertexArray(textVAO);

        // Character size on screen
        float charWidth = CHAR_PIXEL_SIZE * scale;
        float charHeight = CHAR_PIXEL_SIZE * scale;

        // UV size per character in texture
        float uvSize = 1.0f / CHARS_PER_ROW;

        // Render each character
        float cursorX = x;
        for (char c : text) {
            unsigned char ch = static_cast<unsigned char>(c);

            // Handle newlines
            if (ch == '\n') {
                cursorX = x;
                y += charHeight;
                continue;
            }

            // Calculate UV coordinates for this character
            // Characters are arranged in a 16x16 grid in ASCII order
            int charCol = ch % CHARS_PER_ROW;
            int charRow = ch / CHARS_PER_ROW;

            // UV coordinates (flip Y because texture is flipped)
            float u0 = charCol * uvSize;
            float v0 = 1.0f - (charRow + 1) * uvSize;  // Bottom of char
            float u1 = (charCol + 1) * uvSize;
            float v1 = 1.0f - charRow * uvSize;         // Top of char

            // Build quad vertices (position.xy, texcoord.xy)
            float vertices[6][4] = {
                // First triangle (top-left, bottom-left, bottom-right)
                { cursorX,              y,              u0, v1 },
                { cursorX,              y + charHeight, u0, v0 },
                { cursorX + charWidth,  y + charHeight, u1, v0 },
                // Second triangle (top-left, bottom-right, top-right)
                { cursorX,              y,              u0, v1 },
                { cursorX + charWidth,  y + charHeight, u1, v0 },
                { cursorX + charWidth,  y,              u1, v1 },
            };

            // Update VBO and draw
            glBindBuffer(GL_ARRAY_BUFFER, textVBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glDrawArrays(GL_TRIANGLES, 0, 6);

            // Advance cursor
            cursorX += charWidth;
        }

        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);

        // Restore state
        if (depthTestEnabled) glEnable(GL_DEPTH_TEST);
        else glDisable(GL_DEPTH_TEST);

        if (blendEnabled) glEnable(GL_BLEND);
        else glDisable(GL_BLEND);

        glBlendFunc(blendSrc, blendDst);
    }

    void drawTextWithShadow(float x, float y, const std::string& text,
        float scale, const glm::vec4& color,
        const glm::vec4& shadowColor, float shadowOffset) {
        // Draw shadow first (offset down-right)
        drawText(x + shadowOffset, y + shadowOffset, text, scale, shadowColor);
        // Draw main text on top
        drawText(x, y, text, scale, color);
    }

} // namespace TextRenderer