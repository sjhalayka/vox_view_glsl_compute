#ifndef TEXT_RENDERER_H
#define TEXT_RENDERER_H

#include <GL/glew.h>
#include <string>
#include <glm/glm.hpp>

// ============================================================================
// SCREEN-SPACE TEXT RENDERING SYSTEM
// Uses a 1024x1024 bitmap font texture with 16x16 character grid (64x64 per char)
// ============================================================================

namespace TextRenderer {

    // Initialize the text rendering system
    // fontPath: path to the 1024x1024 font bitmap (16x16 character grid)
    // Returns true on success
    bool init(const char* fontPath);

    // Cleanup resources
    void cleanup();

    // Set the screen dimensions (call on window resize)
    void setScreenSize(int width, int height);

    // Render text at screen position (pixels from top-left)
    // x, y: screen position in pixels
    // text: string to render
    // scale: character scale (1.0 = 64 pixels per character)
    // color: RGBA color for the text
    void drawText(float x, float y, const std::string& text,
        float scale = 0.5f,
        const glm::vec4& color = glm::vec4(1.0f));

    // Render text with a shadow for better visibility
    void drawTextWithShadow(float x, float y, const std::string& text,
        float scale = 0.5f,
        const glm::vec4& color = glm::vec4(1.0f),
        const glm::vec4& shadowColor = glm::vec4(0.0f, 0.0f, 0.0f, 0.8f),
        float shadowOffset = 2.0f);

    // Get the width of a string in pixels (at given scale)
    float getTextWidth(const std::string& text, float scale = 0.5f);

    // Get character height in pixels (at given scale)
    float getCharHeight(float scale = 0.5f);

    // Internal state - exposed for advanced usage
    extern GLuint fontTexture;
    extern GLuint textShaderProgram;
    extern GLuint textVAO, textVBO;
    extern int screenWidth, screenHeight;

    // Character dimensions in the font texture
    const int FONT_TEXTURE_SIZE = 1024;
    const int CHARS_PER_ROW = 16;
    const int CHAR_PIXEL_SIZE = FONT_TEXTURE_SIZE / CHARS_PER_ROW;  // 64 pixels

} // namespace TextRenderer

#endif // TEXT_RENDERER_H