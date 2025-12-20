#include "stb_image.h"

#include "main.h"
#include "shader_utils.h"





GLuint loadFontTexture(const char* filename) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Load image using stb_image (you're already using this)
    int width, height, channels;
    stbi_set_flip_vertically_on_load(false);


    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);

    if (!data) {
        std::cerr << "Failed to load font texture: " << filename << std::endl;
        std::cerr << "STB Image error: " << stbi_failure_reason() << std::endl;
        return 0;
    }

    // Determine format based on channels
    GLenum format;
    switch (channels) {
    case 1: format = GL_RED; break;
    case 3: format = GL_RGB; break;
    case 4: format = GL_RGBA; break;
    default:
        format = GL_RGB;
        std::cerr << "Unsupported number of channels: " << channels << std::endl;
    }

    // Load texture data to GPU
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

    // Free image data
    stbi_image_free(data);

    return textureID;
}

FontAtlas initFontAtlas(const char* filename) {
    FontAtlas atlas;
    atlas.textureID = loadFontTexture(filename);
    atlas.charWidth = 64;
    atlas.charHeight = 64;
    atlas.atlasWidth = 1024;
    atlas.atlasHeight = 1024;
    atlas.charsPerRow = atlas.atlasWidth / atlas.charWidth; // 16

    return atlas;
}


class TextRenderer {
private:
    FontAtlas atlas;
    GLuint VAO, VBO, EBO;
    GLuint shaderProgram;
    glm::mat4 projection;

    struct Vertex {
        glm::vec3 position;
        glm::vec2 texCoord;
        glm::vec4 color;
    };

public:

    std::unordered_map<char, int> charWidths; // Map to store actual widths

    // Add this method to calculate character widths
    void calculateCharacterWidths() {
        // Read back the font texture data from GPU
        int dataSize = atlas.atlasWidth * atlas.atlasHeight * 4; // RGBA format
        std::vector<unsigned char> textureData(dataSize);

        glBindTexture(GL_TEXTURE_2D, atlas.textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData.data());





        // Analyze each character
        for (unsigned short c_ = 0; c_ < 256; c_++)
        {
            unsigned char c = static_cast<unsigned char>(c_);

            // ASCII range
            int atlasX = (c % atlas.charsPerRow) * atlas.charWidth;
            int atlasY = (c / atlas.charsPerRow) * atlas.charHeight;

            // Find leftmost non-empty column
            int leftEdge = atlas.charWidth - 1; // Start from rightmost position

            // Find rightmost non-empty column
            int rightEdge = 0; // Start from leftmost position

            // Scan all columns for this character
            for (int x = 0; x < atlas.charWidth; x++) {
                bool columnHasPixels = false;

                // Check if any pixel in this column is non-transparent
                for (int y = 0; y < atlas.charHeight; y++) {
                    int pixelIndex = ((atlasY + y) * atlas.atlasWidth + (atlasX + x)) * 4;
                    if (pixelIndex >= 0 && pixelIndex < dataSize - 3) {
                        // Check alpha value (using red channel for grayscale font)
                        if (textureData[pixelIndex] > 20) { // Non-transparent threshold
                            columnHasPixels = true;
                            break;
                        }
                    }
                }

                if (columnHasPixels) {
                    // Update left edge (minimum value)
                    leftEdge = std::min(leftEdge, x);
                    // Update right edge (maximum value)
                    rightEdge = std::max(rightEdge, x);
                }
            }

            // If no pixels were found (space or empty character)
            if (rightEdge < leftEdge) {
                // Default width for space character
                if (c == ' ') {
                    charWidths[c] = atlas.charWidth / 3; // Make space 1/3 of cell width
                }
                else {
                    charWidths[c] = atlas.charWidth / 4; // Default minimum width
                }
            }
            else {
                // Calculate width based on the actual character bounds
                int actualWidth = (rightEdge - leftEdge) + 1;

                // Add some padding
                int paddedWidth = actualWidth + 4; // 2 pixels on each side

                // Store this character's width (minimum width of 1/4 of the cell)
                charWidths[c] = std::max(paddedWidth, atlas.charWidth / 4);
            }
        }
    }


    TextRenderer(const char* fontAtlasFile, int windowWidth, int windowHeight) {
        // Initialize font atlas
        atlas = initFontAtlas(fontAtlasFile);

        // Create shader program
        GLuint vertexShader = compileShader(GL_VERTEX_SHADER, textVertexShaderSource);
        GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, textFragmentShaderSource);

        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        // Check for linking errors
        GLint success;
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            GLchar infoLog[512];
            glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
            std::cerr << "Shader program linking error: " << infoLog << std::endl;
        }

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        // Create VAO, VBO, EBO for text rendering
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        // Set up vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(glm::vec3)));
        glEnableVertexAttribArray(1);

        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(glm::vec3) + sizeof(glm::vec2)));
        glEnableVertexAttribArray(2);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

        // Set up projection matrix
        setProjection(windowWidth, windowHeight);

        calculateCharacterWidths();
    }

    ~TextRenderer() {
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteVertexArrays(1, &VAO);
        glDeleteProgram(shaderProgram);
        glDeleteTextures(1, &atlas.textureID);
    }

    void setProjection(int windowWidth, int windowHeight) {
        projection = glm::ortho(0.0f, (float)windowWidth, (float)windowHeight, 0.0f, -1.0f, 1.0f);
    }


    void renderText(const std::string& text, float x, float y, float scale, glm::vec4 color, bool centered = false) {
        glUseProgram(shaderProgram);

        // Set uniforms
        GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

        GLuint fontTexLoc = glGetUniformLocation(shaderProgram, "fontTexture");
        glUniform1i(fontTexLoc, 0);

        GLuint useColorLoc = glGetUniformLocation(shaderProgram, "useColor");
        glUniform1i(useColorLoc, 0); // Set to 1 if your font atlas is colored

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, atlas.textureID);

        glBindVertexArray(VAO);

        // If text should be centered, calculate the starting position
        if (centered) {
            float textWidth = 0;
            for (char c : text) {
                // Use the calculated width for each character
                float charWidth = static_cast<float>(charWidths[c]);// charWidths.count(c) ? charWidths[c] : atlas.charWidth / 2;
                textWidth += (8 + charWidth) * scale;
            }
            x = win_x / 2.0f - textWidth / 2.0f;
        }

        // For each character, create a quad with appropriate texture coordinates
        std::vector<Vertex> vertices;
        std::vector<GLuint> indices;

        float xpos = x;
        float ypos = y;
        GLuint indexOffset = 0;

        for (char c : text) {
            // Get ASCII value of the character
            unsigned char charValue = static_cast<unsigned char>(c);

            // Calculate position in the atlas using ASCII value
            int atlasX = (charValue % atlas.charsPerRow) * atlas.charWidth;
            int atlasY = (charValue / atlas.charsPerRow) * atlas.charHeight;

            // Calculate texture coordinates
            float texLeft = atlasX / (float)atlas.atlasWidth;
            float texRight = (atlasX + atlas.charWidth) / (float)atlas.atlasWidth;
            float texTop = atlasY / (float)atlas.atlasHeight;
            float texBottom = (atlasY + atlas.charHeight) / (float)atlas.atlasHeight;

            // Get the character's calculated width
            float charWidth = static_cast<float>(charWidths[charValue]);// charWidths.count(charValue) ? charWidths[charValue] : atlas.charWidth / 2;

            // Calculate quad vertices
            float quadLeft = xpos;
            float quadRight = xpos + atlas.charWidth * scale; // Use full cell width for texture
            float quadTop = ypos;
            float quadBottom = ypos + atlas.charHeight * scale;

            // Add vertices
            vertices.push_back({ {quadLeft, quadTop, 0.0f}, {texLeft, texTop}, color });
            vertices.push_back({ {quadRight, quadTop, 0.0f}, {texRight, texTop}, color });
            vertices.push_back({ {quadRight, quadBottom, 0.0f}, {texRight, texBottom}, color });
            vertices.push_back({ {quadLeft, quadBottom, 0.0f}, {texLeft, texBottom}, color });

            // Add indices
            indices.push_back(indexOffset + 0);
            indices.push_back(indexOffset + 1);
            indices.push_back(indexOffset + 2);
            indices.push_back(indexOffset + 0);
            indices.push_back(indexOffset + 2);
            indices.push_back(indexOffset + 3);

            indexOffset += 4;

            // Advance cursor using the calculated width
            // add 8 pixels of padding between characters
            xpos += (8 + charWidth) * scale;
        }

        // Upload vertex and index data
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_DYNAMIC_DRAW);

        // Draw text
        glm::mat4 model = glm::mat4(1.0f);
        GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

        // Enable blending for transparent font
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);

        // Reset state
        glDisable(GL_BLEND);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
};



TextRenderer* textRenderer = nullptr;

void displayFPS()
{
    static int frame_count = 0;
    static float lastTime = 0.0f;
    static float fps = 0.0f;

    frame_count++;

    float currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
    float deltaTime = currentTime - lastTime;

    if (deltaTime >= 1.0f)
    {
        fps = frame_count / deltaTime;
        frame_count = 0;
        lastTime = currentTime;
    }

    std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));

    if (textRenderer)
    {
        // === CRITICAL FIX: Disable depth test for UI ===
        glDisable(GL_DEPTH_TEST);

        glDisable(GL_CULL_FACE);

        // Optional: also disable depth writes if you have other overlays
        // glDepthMask(GL_FALSE);

        textRenderer->renderText(fpsText, 10.0f, 10.0f, 0.5f, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), true);
        // Or keep centered: textRenderer->renderText(..., true);

        glEnable(GL_CULL_FACE);

        // Re-enable depth test for next 3D rendering (not needed here since display_func ends)
        glEnable(GL_DEPTH_TEST);
    }
}



const char* marchingCubesComputeShader = R"(
#version 430 core

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

// Density field input
layout(std430, binding = 0) readonly buffer DensityField {
    float density[];
};

// Edge table (256 entries)
layout(std430, binding = 1) readonly buffer EdgeTable {
    int edgeTable[];
};

// Triangle table (256 * 16 entries, flattened)
layout(std430, binding = 2) readonly buffer TriTable {
    int triTable[];
};

// Output vertex buffer
layout(std430, binding = 3) buffer VertexBuffer {
    vec4 vertices[];
};

// Output normal buffer
layout(std430, binding = 4) buffer NormalBuffer {
    vec4 normals[];
};

// Atomic counter for vertex count
layout(std430, binding = 5) buffer VertexCounter {
    uint vertexCount;
};

uniform ivec3 gridRes;
uniform vec3 gridMin;
uniform vec3 gridMax;
uniform float isoValue;
uniform uint maxVertices;
uniform uint vertexOffset;  // Offset into vertex buffer for this layer

// Get density at grid position with bounds checking
float getDensity(ivec3 p) {
    p = clamp(p, ivec3(0), gridRes - ivec3(1));
    uint idx = p.x + p.y * gridRes.x + p.z * gridRes.x * gridRes.y;
    return density[idx];
}

// Compute gradient (normal) at a point using central differences
vec3 computeGradient(vec3 worldPos, vec3 cellSize) {
    vec3 gridPos = (worldPos - gridMin) / cellSize;
    ivec3 p = ivec3(floor(gridPos));
    
    float dx = getDensity(p + ivec3(1,0,0)) - getDensity(p - ivec3(1,0,0));
    float dy = getDensity(p + ivec3(0,1,0)) - getDensity(p - ivec3(0,1,0));
    float dz = getDensity(p + ivec3(0,0,1)) - getDensity(p - ivec3(0,0,1));
    
    vec3 grad = vec3(dx, dy, dz);
    float len = length(grad);
    return len > 0.0001 ? grad / len : vec3(0.0, 1.0, 0.0);
}

// Interpolate vertex position along edge
vec3 interpolateVertex(vec3 p1, vec3 p2, float v1, float v2) {
    if (abs(isoValue - v1) < 0.00001) return p1;
    if (abs(isoValue - v2) < 0.00001) return p2;
    if (abs(v1 - v2) < 0.00001) return p1;
    
    float t = (isoValue - v1) / (v2 - v1);
    t = clamp(t, 0.0, 1.0);
    return mix(p1, p2, t);
}

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    // Leave one cell margin for gradient computation
    if (gid.x >= gridRes.x - 1 || gid.y >= gridRes.y - 1 || gid.z >= gridRes.z - 1) {
        return;
    }
    
    vec3 cellSize = (gridMax - gridMin) / vec3(gridRes - ivec3(1));
    
    // Get the 8 corner positions of this cell
    vec3 p[8];
    p[0] = gridMin + vec3(gid) * cellSize;
    p[1] = p[0] + vec3(cellSize.x, 0.0, 0.0);
    p[2] = p[0] + vec3(cellSize.x, cellSize.y, 0.0);
    p[3] = p[0] + vec3(0.0, cellSize.y, 0.0);
    p[4] = p[0] + vec3(0.0, 0.0, cellSize.z);
    p[5] = p[0] + vec3(cellSize.x, 0.0, cellSize.z);
    p[6] = p[0] + vec3(cellSize.x, cellSize.y, cellSize.z);
    p[7] = p[0] + vec3(0.0, cellSize.y, cellSize.z);
    
    // Get density values at corners
    float v[8];
    v[0] = getDensity(gid);
    v[1] = getDensity(gid + ivec3(1, 0, 0));
    v[2] = getDensity(gid + ivec3(1, 1, 0));
    v[3] = getDensity(gid + ivec3(0, 1, 0));
    v[4] = getDensity(gid + ivec3(0, 0, 1));
    v[5] = getDensity(gid + ivec3(1, 0, 1));
    v[6] = getDensity(gid + ivec3(1, 1, 1));
    v[7] = getDensity(gid + ivec3(0, 1, 1));
    
    // Compute cube index (which corners are inside the isosurface)
    int cubeIndex = 0;
    if (v[0] < isoValue) cubeIndex |= 1;
    if (v[1] < isoValue) cubeIndex |= 2;
    if (v[2] < isoValue) cubeIndex |= 4;
    if (v[3] < isoValue) cubeIndex |= 8;
    if (v[4] < isoValue) cubeIndex |= 16;
    if (v[5] < isoValue) cubeIndex |= 32;
    if (v[6] < isoValue) cubeIndex |= 64;
    if (v[7] < isoValue) cubeIndex |= 128;
    
    // Early exit if cube is entirely inside or outside
    int edges = edgeTable[cubeIndex];
    if (edges == 0) return;
    
    // Compute edge vertices where surface intersects
    vec3 edgeVerts[12];
    
    if ((edges & 1) != 0)    edgeVerts[0]  = interpolateVertex(p[0], p[1], v[0], v[1]);
    if ((edges & 2) != 0)    edgeVerts[1]  = interpolateVertex(p[1], p[2], v[1], v[2]);
    if ((edges & 4) != 0)    edgeVerts[2]  = interpolateVertex(p[2], p[3], v[2], v[3]);
    if ((edges & 8) != 0)    edgeVerts[3]  = interpolateVertex(p[3], p[0], v[3], v[0]);
    if ((edges & 16) != 0)   edgeVerts[4]  = interpolateVertex(p[4], p[5], v[4], v[5]);
    if ((edges & 32) != 0)   edgeVerts[5]  = interpolateVertex(p[5], p[6], v[5], v[6]);
    if ((edges & 64) != 0)   edgeVerts[6]  = interpolateVertex(p[6], p[7], v[6], v[7]);
    if ((edges & 128) != 0)  edgeVerts[7]  = interpolateVertex(p[7], p[4], v[7], v[4]);
    if ((edges & 256) != 0)  edgeVerts[8]  = interpolateVertex(p[0], p[4], v[0], v[4]);
    if ((edges & 512) != 0)  edgeVerts[9]  = interpolateVertex(p[1], p[5], v[1], v[5]);
    if ((edges & 1024) != 0) edgeVerts[10] = interpolateVertex(p[2], p[6], v[2], v[6]);
    if ((edges & 2048) != 0) edgeVerts[11] = interpolateVertex(p[3], p[7], v[3], v[7]);
    
    // Generate triangles from the tri table
    int triTableBase = cubeIndex * 16;
    
    for (int i = 0; triTable[triTableBase + i] != -1; i += 3) {
        // Atomically allocate space for 3 vertices
        uint baseIdx = atomicAdd(vertexCount, 3);
        
        // Check bounds
        if (baseIdx + 2 >= maxVertices) return;
        
        // Add offset for this layer
        baseIdx += vertexOffset;
        
        // Get triangle vertex indices
        int e0 = triTable[triTableBase + i];
        int e1 = triTable[triTableBase + i + 1];
        int e2 = triTable[triTableBase + i + 2];
        
        vec3 v0 = edgeVerts[e0];
        vec3 v1 = edgeVerts[e1];
        vec3 v2 = edgeVerts[e2];
        
        // Compute face normal
        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;
        vec3 faceNormal = normalize(cross(edge1, edge2));
        
        // Store vertices and normals
        vertices[baseIdx]     = vec4(v0, 1.0);
        vertices[baseIdx + 1] = vec4(v1, 1.0);
        vertices[baseIdx + 2] = vec4(v2, 1.0);
        
        // Use gradient-based normals for smoother shading
        normals[baseIdx]     = vec4(computeGradient(v0, cellSize), 0.0);
        normals[baseIdx + 1] = vec4(computeGradient(v1, cellSize), 0.0);
        normals[baseIdx + 2] = vec4(computeGradient(v2, cellSize), 0.0);
    }
}
)";

// ============================================================================
// MARCHING CUBES - Render Shaders (with transparency and lighting)
// ============================================================================

const char* mcVertexShaderSource = R"(
#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragPos;
out vec3 fragNormal;

void main() {
    vec4 worldPos = model * vec4(position, 1.0);
    fragPos = worldPos.xyz;
    fragNormal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * worldPos;
}
)";

const char* mcFragmentShaderSource = R"(
#version 430 core

in vec3 fragPos;
in vec3 fragNormal;

uniform vec3 viewPos;
uniform vec4 layerColor;  // RGB + alpha
uniform vec3 lightDir;

out vec4 FragColor;

void main() {

    FragColor = layerColor;
    return;

    //vec3 normal = normalize(fragNormal);

    //// Simple directional lighting
    //vec3 lightColor = vec3(1.0);
    //float ambientStrength = 0.3;
    //vec3 ambient = ambientStrength * lightColor;
    //
    //// Diffuse - use abs for double-sided lighting
    //float diff = abs(dot(normal, normalize(-lightDir)));
    //vec3 diffuse = diff * lightColor;
    //
    //// Specular
    //vec3 viewDir = normalize(viewPos - fragPos);
    //vec3 reflectDir = reflect(normalize(lightDir), normal);
    //float spec = pow(max(abs(dot(viewDir, reflectDir)), 0.0), 32.0);
    //vec3 specular = 0.3 * spec * lightColor;
    //
    //vec3 lighting = ambient + diffuse + specular;
    //vec3 finalColor = lighting * layerColor.rgb;
    //
    //FragColor = vec4(finalColor, layerColor.a);
}
)";









const char* backgroundPointsComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Voxel object data
layout(std430, binding = 0) readonly buffer VoxelCentres {
    vec4 voxelCentres[];  // xyz = position, w = unused
};

layout(std430, binding = 1) readonly buffer VoxelDensities {
    float voxelDensities[];
};

layout(std430, binding = 2) readonly buffer GridMinMax {
    vec4 gridMin;      // xyz = min, w = cell_size
    vec4 gridMax;      // xyz = max, w = unused
    ivec4 voxelRes;    // xyz = resolution, w = unused
};

// Output buffers
layout(std430, binding = 3) writeonly buffer BackgroundDensities {
    float backgroundDensities[];
};

layout(std430, binding = 4) writeonly buffer BackgroundCollisions {
    int backgroundCollisions[];
};

// Grid cells lookup table (maps cell position to voxel index)
layout(std430, binding = 6) readonly buffer VoGridCells {
    int voGridCells[];
};

// Uniforms
uniform mat4 invModelMatrix;
uniform vec3 bgGridMin;
uniform vec3 bgGridMax;
uniform ivec3 bgRes;

// Find voxel containing a point (in local/model space)
int findVoxelContainingPoint(vec3 point) {
    float cellSize = gridMin.w;
    vec3 gMin = gridMin.xyz;
    
    // Get grid cell coordinates
    ivec3 cell = ivec3(floor((point - gMin) / cellSize));
    
    // Check bounds
    if (cell.x < 0 || cell.x >= voxelRes.x ||
        cell.y < 0 || cell.y >= voxelRes.y ||
        cell.z < 0 || cell.z >= voxelRes.z) {
        return -1;
    }
    
    // Find the index in the flattened 3D array
    int cellIndex = cell.x + cell.y * voxelRes.x + cell.z * voxelRes.x * voxelRes.y;
    
    if (cellIndex < 0 || cellIndex >= voGridCells.length()) {
        return -1;
    }
    
    // Use the grid cells lookup to get actual voxel index
    int voxelIdx = voGridCells[cellIndex];
    
    if (voxelIdx == -1) {
        return -1;
    }
    
    // Precise check against the voxel
    float halfSize = cellSize * 0.5;
    vec3 center = voxelCentres[voxelIdx].xyz;
    
    if (point.x >= center.x - halfSize && point.x <= center.x + halfSize &&
        point.y >= center.y - halfSize && point.y <= center.y + halfSize &&
        point.z >= center.z - halfSize && point.z <= center.z + halfSize) {
        return voxelIdx;
    }
    
    return -1;
}

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= bgRes.x || gid.y >= bgRes.y || gid.z >= bgRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * bgRes.x + gid.z * bgRes.x * bgRes.y;
    
    // Calculate world position for this grid point
    vec3 stepSize = (bgGridMax - bgGridMin) / vec3(bgRes - 1);
    vec3 worldPos = bgGridMin + vec3(gid) * stepSize;
    
    // Transform to local space using inverse model matrix
    vec4 localPos = invModelMatrix * vec4(worldPos, 1.0);
    
    // Check if point is inside any voxel
    int voxelIndex = findVoxelContainingPoint(localPos.xyz);
    
    // Accumulate results (union of all obstacles)
    // Only set to 1 if we found a voxel, never clear it back to 0
    if (voxelIndex >= 0) {
        backgroundDensities[index] = 1.0;
        backgroundCollisions[index] = voxelIndex;
    }
    // Note: We don't set to 0 here - the buffer is pre-cleared before processing all objects
}
)";

const char* surfaceDetectionComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 3) readonly buffer BackgroundDensities {
    float backgroundDensities[];
};

layout(std430, binding = 5) writeonly buffer SurfaceDensities {
    float surfaceDensities[];
};

uniform ivec3 bgRes;

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= bgRes.x || gid.y >= bgRes.y || gid.z >= bgRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * bgRes.x + gid.z * bgRes.x * bgRes.y;
    
    // Skip points that are inside the voxel grid
    if (backgroundDensities[index] > 0.0) {
        surfaceDensities[index] = 0.0;
        return;
    }
    
    // Check 6 neighbors for surface detection
    const ivec3 dirs[6] = ivec3[6](
        ivec3(1, 0, 0), ivec3(-1, 0, 0),
        ivec3(0, 1, 0), ivec3(0, -1, 0),
        ivec3(0, 0, 1), ivec3(0, 0, -1)
    );
    
    bool isSurface = false;
    
    for (int i = 0; i < 6; i++) {
        ivec3 neighbor = gid + dirs[i];
        
        if (neighbor.x < 0 || neighbor.x >= bgRes.x ||
            neighbor.y < 0 || neighbor.y >= bgRes.y ||
            neighbor.z < 0 || neighbor.z >= bgRes.z) {
            continue;
        }
        
        uint neighborIndex = neighbor.x + neighbor.y * bgRes.x + neighbor.z * bgRes.x * bgRes.y;
        
        if (backgroundDensities[neighborIndex] > 0.0) {
            isSurface = true;
            break;
        }
    }
    
    surfaceDensities[index] = isSurface ? 1.0 : 0.0;
}
)";

// ============================================================================
// FLUID SIMULATION - Compute Shaders
// ============================================================================

// Update obstacle mask from voxel collisions
const char* obstacleComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) readonly buffer BackgroundDensities {
    float backgroundDensities[];
};

layout(std430, binding = 1) writeonly buffer Obstacles {
    float obstacles[];
};

uniform ivec3 gridRes;

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= gridRes.x || gid.y >= gridRes.y || gid.z >= gridRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y;
    
    // Point is an obstacle if it's inside the voxel grid
    obstacles[index] = backgroundDensities[index] > 0.0 ? 1.0 : 0.0;
}
)";

// Advection shader (semi-Lagrangian)
const char* advectionComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) readonly buffer VelocityIn {
    vec4 velocityIn[];
};

layout(std430, binding = 1) readonly buffer FieldIn {
    float fieldIn[];
};

layout(std430, binding = 2) writeonly buffer FieldOut {
    float fieldOut[];
};

layout(std430, binding = 3) readonly buffer Obstacles {
    float obstacles[];
};

uniform ivec3 gridRes;
uniform vec3 gridMin;
uniform vec3 gridMax;
uniform float dt;
uniform float dissipation;

// Trilinear interpolation
float sampleField(vec3 pos) {
    vec3 gridSize = vec3(gridRes);
    vec3 cellSize = (gridMax - gridMin) / (gridSize - 1.0);
    
    // Convert world position to grid coordinates
    vec3 gridPos = (pos - gridMin) / cellSize;
    gridPos = clamp(gridPos, vec3(0.5), gridSize - vec3(1.5));
    
    ivec3 i0 = ivec3(floor(gridPos));
    ivec3 i1 = i0 + ivec3(1);
    
    i0 = clamp(i0, ivec3(0), gridRes - ivec3(1));
    i1 = clamp(i1, ivec3(0), gridRes - ivec3(1));
    
    vec3 t = fract(gridPos);
    
    // 8 corner samples
    float c000 = fieldIn[i0.x + i0.y * gridRes.x + i0.z * gridRes.x * gridRes.y];
    float c100 = fieldIn[i1.x + i0.y * gridRes.x + i0.z * gridRes.x * gridRes.y];
    float c010 = fieldIn[i0.x + i1.y * gridRes.x + i0.z * gridRes.x * gridRes.y];
    float c110 = fieldIn[i1.x + i1.y * gridRes.x + i0.z * gridRes.x * gridRes.y];
    float c001 = fieldIn[i0.x + i0.y * gridRes.x + i1.z * gridRes.x * gridRes.y];
    float c101 = fieldIn[i1.x + i0.y * gridRes.x + i1.z * gridRes.x * gridRes.y];
    float c011 = fieldIn[i0.x + i1.y * gridRes.x + i1.z * gridRes.x * gridRes.y];
    float c111 = fieldIn[i1.x + i1.y * gridRes.x + i1.z * gridRes.x * gridRes.y];
    
    // Trilinear interpolation
    float c00 = mix(c000, c100, t.x);
    float c10 = mix(c010, c110, t.x);
    float c01 = mix(c001, c101, t.x);
    float c11 = mix(c011, c111, t.x);
    
    float c0 = mix(c00, c10, t.y);
    float c1 = mix(c01, c11, t.y);
    
    return mix(c0, c1, t.z);
}

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= gridRes.x || gid.y >= gridRes.y || gid.z >= gridRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y;
    
    // Skip obstacles
    if (obstacles[index] > 0.5) {
        fieldOut[index] = 0.0;
        return;
    }
    
    // Calculate world position
    vec3 cellSize = (gridMax - gridMin) / vec3(gridRes - 1);
    vec3 pos = gridMin + vec3(gid) * cellSize;
    
    // Get velocity at this position
    vec3 vel = velocityIn[index].xyz;
    
    // Trace back in time (semi-Lagrangian)
    vec3 prevPos = pos - vel * dt;
    
    // Sample the field at the previous position
    float value = sampleField(prevPos);
    
    // Apply dissipation
    fieldOut[index] = value * dissipation;
}
)";

// Velocity advection shader (advects velocity field itself)
const char* velocityAdvectionComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) readonly buffer VelocityIn {
    vec4 velocityIn[];
};

layout(std430, binding = 1) writeonly buffer VelocityOut {
    vec4 velocityOut[];
};

layout(std430, binding = 2) readonly buffer Obstacles {
    float obstacles[];
};

uniform ivec3 gridRes;
uniform vec3 gridMin;
uniform vec3 gridMax;
uniform float dt;
uniform float dissipation;

// Trilinear interpolation for velocity
vec3 sampleVelocity(vec3 pos) {
    vec3 gridSize = vec3(gridRes);
    vec3 cellSize = (gridMax - gridMin) / (gridSize - 1.0);
    
    vec3 gridPos = (pos - gridMin) / cellSize;
    gridPos = clamp(gridPos, vec3(0.5), gridSize - vec3(1.5));
    
    ivec3 i0 = ivec3(floor(gridPos));
    ivec3 i1 = i0 + ivec3(1);
    
    i0 = clamp(i0, ivec3(0), gridRes - ivec3(1));
    i1 = clamp(i1, ivec3(0), gridRes - ivec3(1));
    
    vec3 t = fract(gridPos);
    
    vec3 c000 = velocityIn[i0.x + i0.y * gridRes.x + i0.z * gridRes.x * gridRes.y].xyz;
    vec3 c100 = velocityIn[i1.x + i0.y * gridRes.x + i0.z * gridRes.x * gridRes.y].xyz;
    vec3 c010 = velocityIn[i0.x + i1.y * gridRes.x + i0.z * gridRes.x * gridRes.y].xyz;
    vec3 c110 = velocityIn[i1.x + i1.y * gridRes.x + i0.z * gridRes.x * gridRes.y].xyz;
    vec3 c001 = velocityIn[i0.x + i0.y * gridRes.x + i1.z * gridRes.x * gridRes.y].xyz;
    vec3 c101 = velocityIn[i1.x + i0.y * gridRes.x + i1.z * gridRes.x * gridRes.y].xyz;
    vec3 c011 = velocityIn[i0.x + i1.y * gridRes.x + i1.z * gridRes.x * gridRes.y].xyz;
    vec3 c111 = velocityIn[i1.x + i1.y * gridRes.x + i1.z * gridRes.x * gridRes.y].xyz;
    
    vec3 c00 = mix(c000, c100, t.x);
    vec3 c10 = mix(c010, c110, t.x);
    vec3 c01 = mix(c001, c101, t.x);
    vec3 c11 = mix(c011, c111, t.x);
    
    vec3 c0 = mix(c00, c10, t.y);
    vec3 c1 = mix(c01, c11, t.y);
    
    return mix(c0, c1, t.z);
}

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= gridRes.x || gid.y >= gridRes.y || gid.z >= gridRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y;
    
    // Skip obstacles
    if (obstacles[index] > 0.5) {
        velocityOut[index] = vec4(0.0);
        return;
    }
    
    vec3 cellSize = (gridMax - gridMin) / vec3(gridRes - 1);
    vec3 pos = gridMin + vec3(gid) * cellSize;
    
    vec3 vel = velocityIn[index].xyz;
    vec3 prevPos = pos - vel * dt;
    
    vec3 newVel = sampleVelocity(prevPos);
    velocityOut[index] = vec4(newVel * dissipation, 0.0);
}
)";

// ============================================================================
// NEW: Buoyancy and Gravity Forces Compute Shader
// ============================================================================
const char* buoyancyComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) buffer Velocity {
    vec4 velocity[];
};

layout(std430, binding = 1) readonly buffer Density {
    float density[];
};

layout(std430, binding = 2) readonly buffer Temperature {
    float temperature[];
};

layout(std430, binding = 3) readonly buffer Obstacles {
    float obstacles[];
};

uniform ivec3 gridRes;
uniform float dt;
uniform float ambientTemperature;
uniform float buoyancyAlpha;  // Density coefficient (makes dense fluid sink)
uniform float buoyancyBeta;   // Temperature coefficient (makes hot fluid rise)
uniform float gravity;        // Gravity magnitude
uniform vec3 gravityDirection; // Gravity direction (usually (0, -1, 0))
uniform bool enableGravity;
uniform bool enableBuoyancy;

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= gridRes.x || gid.y >= gridRes.y || gid.z >= gridRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y;
    
    // Skip obstacles
    if (obstacles[index] > 0.5) {
        return;
    }
    
    vec3 vel = velocity[index].xyz;
    vec3 force = vec3(0.0);
    
    // Up direction is opposite of gravity direction
    vec3 upDirection = -normalize(gravityDirection);
    
    // Gravity: constant downward force
    // F_gravity = g * gravityDirection
    if (enableGravity) {
        force += gravity * gravityDirection;
    }
    
    // Buoyancy force (Boussinesq approximation)
    // F_buoy = (-alpha * density + beta * (T - T_ambient)) * up
    // - Dense fluid sinks (alpha term, negative contribution to upward force)
    // - Hot fluid rises (beta term, positive contribution to upward force)
    if (enableBuoyancy) {
        float d = density[index];
        float T = temperature[index];
        
        // Buoyancy: hot air rises, dense smoke sinks
        float buoyancyForce = -buoyancyAlpha * d + buoyancyBeta * (T - ambientTemperature);
        force += buoyancyForce * upDirection;
    }
    
    // Apply forces: v += F * dt
    vel += force * dt;
    
    velocity[index] = vec4(vel, 0.0);
}
)";

// Smagorinsky LES turbulence model
const char* turbulenceComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) readonly buffer VelocityIn {
    vec4 velocityIn[];
};

layout(std430, binding = 1) buffer VelocityOut {
    vec4 velocityOut[];
};

layout(std430, binding = 2) readonly buffer Obstacles {
    float obstacles[];
};

layout(std430, binding = 3) writeonly buffer TurbulentViscosity {
    float turbulentViscosity[];
};

uniform ivec3 gridRes;
uniform vec3 gridMin;
uniform vec3 gridMax;
uniform float dt;
uniform float smagorinskyConst;  // Typically 0.1 - 0.2
uniform float baseViscosity;

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= gridRes.x || gid.y >= gridRes.y || gid.z >= gridRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y;
    
    if (obstacles[index] > 0.5) {
        velocityOut[index] = vec4(0.0);
        turbulentViscosity[index] = 0.0;
        return;
    }
    
    vec3 cellSize = (gridMax - gridMin) / vec3(gridRes - 1);
    float dx = cellSize.x;
    
    // Get neighboring velocities for gradient calculation
    // Boundary handling
    int xm = max(gid.x - 1, 0);
    int xp = min(gid.x + 1, gridRes.x - 1);
    int ym = max(gid.y - 1, 0);
    int yp = min(gid.y + 1, gridRes.y - 1);
    int zm = max(gid.z - 1, 0);
    int zp = min(gid.z + 1, gridRes.z - 1);
    
    vec3 vxm = velocityIn[xm + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y].xyz;
    vec3 vxp = velocityIn[xp + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y].xyz;
    vec3 vym = velocityIn[gid.x + ym * gridRes.x + gid.z * gridRes.x * gridRes.y].xyz;
    vec3 vyp = velocityIn[gid.x + yp * gridRes.x + gid.z * gridRes.x * gridRes.y].xyz;
    vec3 vzm = velocityIn[gid.x + gid.y * gridRes.x + zm * gridRes.x * gridRes.y].xyz;
    vec3 vzp = velocityIn[gid.x + gid.y * gridRes.x + zp * gridRes.x * gridRes.y].xyz;
    
    // Compute strain rate tensor components (symmetric part of velocity gradient)
    // S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    float dudx = (vxp.x - vxm.x) / (2.0 * dx);
    float dvdy = (vyp.y - vym.y) / (2.0 * dx);
    float dwdz = (vzp.z - vzm.z) / (2.0 * dx);
    
    float dudy = (vyp.x - vym.x) / (2.0 * dx);
    float dvdx = (vxp.y - vxm.y) / (2.0 * dx);
    
    float dudz = (vzp.x - vzm.x) / (2.0 * dx);
    float dwdx = (vxp.z - vxm.z) / (2.0 * dx);
    
    float dvdz = (vzp.y - vzm.y) / (2.0 * dx);
    float dwdy = (vyp.z - vym.z) / (2.0 * dx);
    
    // Strain rate magnitude |S| = sqrt(2 * S_ij * S_ij)
    float S11 = dudx;
    float S22 = dvdy;
    float S33 = dwdz;
    float S12 = 0.5 * (dudy + dvdx);
    float S13 = 0.5 * (dudz + dwdx);
    float S23 = 0.5 * (dvdz + dwdy);
    
    float S_magnitude = sqrt(2.0 * (S11*S11 + S22*S22 + S33*S33 + 2.0*(S12*S12 + S13*S13 + S23*S23)));
    
    // Smagorinsky model: nu_t = (Cs * delta)^2 * |S|
    float delta = dx;  // Filter width = grid spacing
    float nu_t = pow(smagorinskyConst * delta, 2.0) * S_magnitude;
    
    turbulentViscosity[index] = nu_t;
    
    // Apply turbulent diffusion to velocity
    // Laplacian of velocity
    vec3 vel = velocityIn[index].xyz;
    vec3 laplacian = (vxp + vxm + vyp + vym + vzp + vzm - 6.0 * vel) / (dx * dx);
    
    // Total viscosity = base + turbulent
    float totalViscosity = baseViscosity + nu_t;
    
    // Explicit diffusion step
    vec3 newVel = vel + totalViscosity * laplacian * dt;
    
    velocityOut[index] = vec4(newVel, 0.0);
}
)";

// Diffusion solver (Jacobi iteration)
const char* diffusionComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) readonly buffer FieldIn {
    float fieldIn[];
};

layout(std430, binding = 1) buffer FieldOut {
    float fieldOut[];
};

layout(std430, binding = 2) readonly buffer Obstacles {
    float obstacles[];
};

uniform ivec3 gridRes;
uniform float alpha;  // dx^2 / (viscosity * dt)
uniform float rBeta;  // 1 / (6 + alpha)

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= gridRes.x || gid.y >= gridRes.y || gid.z >= gridRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y;
    
    if (obstacles[index] > 0.5) {
        fieldOut[index] = 0.0;
        return;
    }
    
    // Get neighbors with boundary handling
    int xm = max(gid.x - 1, 0);
    int xp = min(gid.x + 1, gridRes.x - 1);
    int ym = max(gid.y - 1, 0);
    int yp = min(gid.y + 1, gridRes.y - 1);
    int zm = max(gid.z - 1, 0);
    int zp = min(gid.z + 1, gridRes.z - 1);
    
    float xmVal = fieldIn[xm + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y];
    float xpVal = fieldIn[xp + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y];
    float ymVal = fieldIn[gid.x + ym * gridRes.x + gid.z * gridRes.x * gridRes.y];
    float ypVal = fieldIn[gid.x + yp * gridRes.x + gid.z * gridRes.x * gridRes.y];
    float zmVal = fieldIn[gid.x + gid.y * gridRes.x + zm * gridRes.x * gridRes.y];
    float zpVal = fieldIn[gid.x + gid.y * gridRes.x + zp * gridRes.x * gridRes.y];
    
    float center = fieldIn[index];
    
    // Jacobi iteration
    fieldOut[index] = (xmVal + xpVal + ymVal + ypVal + zmVal + zpVal + alpha * center) * rBeta;
}
)";

// Divergence calculation
const char* divergenceComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) readonly buffer Velocity {
    vec4 velocity[];
};

layout(std430, binding = 1) writeonly buffer Divergence {
    float divergence[];
};

layout(std430, binding = 2) readonly buffer Obstacles {
    float obstacles[];
};

uniform ivec3 gridRes;
uniform vec3 gridMin;
uniform vec3 gridMax;

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= gridRes.x || gid.y >= gridRes.y || gid.z >= gridRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y;
    
    if (obstacles[index] > 0.5) {
        divergence[index] = 0.0;
        return;
    }
    
    vec3 cellSize = (gridMax - gridMin) / vec3(gridRes - 1);
    float dx = cellSize.x;
    
    // Get neighbor indices with boundary handling
    int xm = max(gid.x - 1, 0);
    int xp = min(gid.x + 1, gridRes.x - 1);
    int ym = max(gid.y - 1, 0);
    int yp = min(gid.y + 1, gridRes.y - 1);
    int zm = max(gid.z - 1, 0);
    int zp = min(gid.z + 1, gridRes.z - 1);
    
    vec3 vxm = velocity[xm + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y].xyz;
    vec3 vxp = velocity[xp + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y].xyz;
    vec3 vym = velocity[gid.x + ym * gridRes.x + gid.z * gridRes.x * gridRes.y].xyz;
    vec3 vyp = velocity[gid.x + yp * gridRes.x + gid.z * gridRes.x * gridRes.y].xyz;
    vec3 vzm = velocity[gid.x + gid.y * gridRes.x + zm * gridRes.x * gridRes.y].xyz;
    vec3 vzp = velocity[gid.x + gid.y * gridRes.x + zp * gridRes.x * gridRes.y].xyz;
    
    // Handle obstacle boundaries (no-slip)
    if (obstacles[xm + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y] > 0.5) vxm = vec3(0.0);
    if (obstacles[xp + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y] > 0.5) vxp = vec3(0.0);
    if (obstacles[gid.x + ym * gridRes.x + gid.z * gridRes.x * gridRes.y] > 0.5) vym = vec3(0.0);
    if (obstacles[gid.x + yp * gridRes.x + gid.z * gridRes.x * gridRes.y] > 0.5) vyp = vec3(0.0);
    if (obstacles[gid.x + gid.y * gridRes.x + zm * gridRes.x * gridRes.y] > 0.5) vzm = vec3(0.0);
    if (obstacles[gid.x + gid.y * gridRes.x + zp * gridRes.x * gridRes.y] > 0.5) vzp = vec3(0.0);
    
    // Central difference divergence
    float div = ((vxp.x - vxm.x) + (vyp.y - vym.y) + (vzp.z - vzm.z)) / (2.0 * dx);
    
    divergence[index] = div;
}
)";

// Pressure solver (Jacobi iteration)
const char* pressureComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) readonly buffer PressureIn {
    float pressureIn[];
};

layout(std430, binding = 1) writeonly buffer PressureOut {
    float pressureOut[];
};

layout(std430, binding = 2) readonly buffer Divergence {
    float divergence[];
};

layout(std430, binding = 3) readonly buffer Obstacles {
    float obstacles[];
};

uniform ivec3 gridRes;
uniform vec3 gridMin;
uniform vec3 gridMax;

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= gridRes.x || gid.y >= gridRes.y || gid.z >= gridRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y;
    
    if (obstacles[index] > 0.5) {
        pressureOut[index] = 0.0;
        return;
    }
    
    vec3 cellSize = (gridMax - gridMin) / vec3(gridRes - 1);
    float dx = cellSize.x;
    
    // Get neighbors with boundary handling
    int xm = max(gid.x - 1, 0);
    int xp = min(gid.x + 1, gridRes.x - 1);
    int ym = max(gid.y - 1, 0);
    int yp = min(gid.y + 1, gridRes.y - 1);
    int zm = max(gid.z - 1, 0);
    int zp = min(gid.z + 1, gridRes.z - 1);
    
    float pxm = pressureIn[xm + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y];
    float pxp = pressureIn[xp + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y];
    float pym = pressureIn[gid.x + ym * gridRes.x + gid.z * gridRes.x * gridRes.y];
    float pyp = pressureIn[gid.x + yp * gridRes.x + gid.z * gridRes.x * gridRes.y];
    float pzm = pressureIn[gid.x + gid.y * gridRes.x + zm * gridRes.x * gridRes.y];
    float pzp = pressureIn[gid.x + gid.y * gridRes.x + zp * gridRes.x * gridRes.y];
    
    // Handle obstacles - use center pressure for Neumann boundary
    float pCenter = pressureIn[index];
    if (obstacles[xm + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y] > 0.5) pxm = pCenter;
    if (obstacles[xp + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y] > 0.5) pxp = pCenter;
    if (obstacles[gid.x + ym * gridRes.x + gid.z * gridRes.x * gridRes.y] > 0.5) pym = pCenter;
    if (obstacles[gid.x + yp * gridRes.x + gid.z * gridRes.x * gridRes.y] > 0.5) pyp = pCenter;
    if (obstacles[gid.x + gid.y * gridRes.x + zm * gridRes.x * gridRes.y] > 0.5) pzm = pCenter;
    if (obstacles[gid.x + gid.y * gridRes.x + zp * gridRes.x * gridRes.y] > 0.5) pzp = pCenter;
    
    // Jacobi iteration for pressure Poisson equation
    float div = divergence[index];
    pressureOut[index] = (pxm + pxp + pym + pyp + pzm + pzp - dx * dx * div) / 6.0;
}
)";

// Gradient subtraction (pressure projection)
const char* gradientSubtractComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) buffer Velocity {
    vec4 velocity[];
};

layout(std430, binding = 1) readonly buffer Pressure {
    float pressure[];
};

layout(std430, binding = 2) readonly buffer Obstacles {
    float obstacles[];
};

uniform ivec3 gridRes;
uniform vec3 gridMin;
uniform vec3 gridMax;

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= gridRes.x || gid.y >= gridRes.y || gid.z >= gridRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y;
    
    if (obstacles[index] > 0.5) {
        velocity[index] = vec4(0.0);
        return;
    }
    
    vec3 cellSize = (gridMax - gridMin) / vec3(gridRes - 1);
    float dx = cellSize.x;
    
    // Get neighbors
    int xm = max(gid.x - 1, 0);
    int xp = min(gid.x + 1, gridRes.x - 1);
    int ym = max(gid.y - 1, 0);
    int yp = min(gid.y + 1, gridRes.y - 1);
    int zm = max(gid.z - 1, 0);
    int zp = min(gid.z + 1, gridRes.z - 1);
    
    float pxm = pressure[xm + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y];
    float pxp = pressure[xp + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y];
    float pym = pressure[gid.x + ym * gridRes.x + gid.z * gridRes.x * gridRes.y];
    float pyp = pressure[gid.x + yp * gridRes.x + gid.z * gridRes.x * gridRes.y];
    float pzm = pressure[gid.x + gid.y * gridRes.x + zm * gridRes.x * gridRes.y];
    float pzp = pressure[gid.x + gid.y * gridRes.x + zp * gridRes.x * gridRes.y];
    
    // Handle obstacle boundaries
    float pCenter = pressure[index];
    if (obstacles[xm + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y] > 0.5) pxm = pCenter;
    if (obstacles[xp + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y] > 0.5) pxp = pCenter;
    if (obstacles[gid.x + ym * gridRes.x + gid.z * gridRes.x * gridRes.y] > 0.5) pym = pCenter;
    if (obstacles[gid.x + yp * gridRes.x + gid.z * gridRes.x * gridRes.y] > 0.5) pyp = pCenter;
    if (obstacles[gid.x + gid.y * gridRes.x + zm * gridRes.x * gridRes.y] > 0.5) pzm = pCenter;
    if (obstacles[gid.x + gid.y * gridRes.x + zp * gridRes.x * gridRes.y] > 0.5) pzp = pCenter;
    
    // Compute pressure gradient
    vec3 gradient;
    gradient.x = (pxp - pxm) / (2.0 * dx);
    gradient.y = (pyp - pym) / (2.0 * dx);
    gradient.z = (pzp - pzm) / (2.0 * dx);
    
    // Subtract gradient from velocity
    vec3 vel = velocity[index].xyz;
    vel -= gradient;
    
    velocity[index] = vec4(vel, 0.0);
}
)";

// ============================================================================
// MODIFIED: Add density/velocity/temperature source
// ============================================================================
const char* addSourceComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) buffer Density {
    float density[];
};

layout(std430, binding = 1) buffer Velocity {
    vec4 velocity[];
};

layout(std430, binding = 2) readonly buffer Obstacles {
    float obstacles[];
};

layout(std430, binding = 3) buffer Temperature {
    float temperature[];
};

uniform ivec3 gridRes;
uniform vec3 gridMin;
uniform vec3 gridMax;
uniform vec3 sourcePos;
uniform vec3 sourceVelocity;
uniform float sourceRadius;
uniform float densityAmount;
uniform float velocityAmount;
uniform float temperatureAmount;
uniform bool addDensity;
uniform bool addVelocity;
uniform bool addTemperature;

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= gridRes.x || gid.y >= gridRes.y || gid.z >= gridRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y;
    
    // Skip obstacles
    if (obstacles[index] > 0.5) {
        return;
    }
    
    // Calculate world position
    vec3 cellSize = (gridMax - gridMin) / vec3(gridRes - 1);
    vec3 pos = gridMin + vec3(gid) * cellSize;
    
    // Check if within source radius
    float dist = length(pos - sourcePos);
    if (dist < sourceRadius) {
        float falloff = 1.0 - (dist / sourceRadius);
        falloff = falloff * falloff;  // Quadratic falloff
        
        if (addDensity) {
            density[index] += densityAmount * falloff;
        }
        
        if (addVelocity) {
            vec3 vel = velocity[index].xyz;
            vel += sourceVelocity * velocityAmount * falloff;
            velocity[index] = vec4(vel, 0.0);
        }
        
        // NEW: Add temperature with density injection
        if (addTemperature) {
            temperature[index] += temperatureAmount * falloff;
        }
    }
}
)";

// Boundary conditions shader
const char* boundaryComputeShader = R"(
#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) buffer Velocity {
    vec4 velocity[];
};

layout(std430, binding = 1) buffer Density {
    float density[];
};

layout(std430, binding = 2) readonly buffer Obstacles {
    float obstacles[];
};

layout(std430, binding = 3) buffer Temperature {
    float temperature[];
};

uniform ivec3 gridRes;
uniform float ambientTemperature;

void main() {
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    
    if (gid.x >= gridRes.x || gid.y >= gridRes.y || gid.z >= gridRes.z) {
        return;
    }
    
    uint index = gid.x + gid.y * gridRes.x + gid.z * gridRes.x * gridRes.y;
    
    // Zero velocity and density at obstacles
    if (obstacles[index] > 0.5) {
        velocity[index] = vec4(0.0);
        density[index] = 0.0;
        temperature[index] = ambientTemperature;
        return;
    }
    
    // Domain boundary conditions (closed box)
    vec3 vel = velocity[index].xyz;
    
    //// X boundaries
    //if (gid.x == 0) vel.x = max(vel.x, 0.0);
    //if (gid.x == gridRes.x - 1) vel.x = min(vel.x, 0.0);
    //
    //// Y boundaries
    //if (gid.y == 0) vel.y = max(vel.y, 0.0);
    //if (gid.y == gridRes.y - 1) vel.y = min(vel.y, 0.0);
    //
    //// Z boundaries
    //if (gid.z == 0) vel.z = max(vel.z, 0.0);
    //if (gid.z == gridRes.z - 1) vel.z = min(vel.z, 0.0);
    
    velocity[index] = vec4(vel, 0.0);
}
)";



const char* volumeVertSource = R"(
#version 430 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 vTexCoord;

void main()
{
    vTexCoord = aTexCoord;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";



const char* volumeFragSource = R"(
#version 430 core
out vec4 FragColor;
in vec2 vTexCoord;

uniform sampler3D densityTex;
uniform sampler3D temperatureTex;
uniform sampler3D obstacleTex;

uniform mat4 invViewProj;
uniform vec3 cameraPos;
uniform vec3 gridMin;
uniform vec3 gridMax;
uniform ivec3 gridRes;

uniform bool visualizeTemperature;
uniform float densityFactor;
uniform float temperatureThreshold;
uniform float stepSize;
uniform int maxSteps;

// ============================================================================
// POINT LIGHT UNIFORMS (for shadow cubemaps)
// ============================================================================
#define MAX_POINT_LIGHTS 8

uniform int numPointLights;
uniform vec3 lightPositions[MAX_POINT_LIGHTS];
uniform float lightIntensities[MAX_POINT_LIGHTS];
uniform vec3 lightColors[MAX_POINT_LIGHTS];
uniform float lightFarPlanes[MAX_POINT_LIGHTS];
uniform int lightEnabled[MAX_POINT_LIGHTS];

uniform samplerCube shadowMaps[MAX_POINT_LIGHTS];


// ============================================================================
// VOLUME LIGHTING PARAMETERS
// ============================================================================
uniform float volumeAbsorption;
uniform float volumeScattering;
uniform int shadowSamples;
uniform float shadowDensityScale;
uniform float phaseG;              // Phase function asymmetry: -1 to 1
uniform bool enableVolumeShadows;
uniform bool enableVolumeLighting;
uniform vec3 ambientLight;

// ============================================================================
// HENYEY-GREENSTEIN PHASE FUNCTION
// ============================================================================
float phaseHG(float cosTheta, float g) {
    if (abs(g) < 0.001) return 1.0;  // Isotropic scattering
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * 3.14159 * pow(denom, 1.5));
}

// ============================================================================
// SHADOW CUBEMAP SAMPLING (for solid geometry shadows)
// ============================================================================
float sampleVolumeShadow(int lightIndex, vec3 worldPos) {
    vec3 fragToLight = worldPos - lightPositions[lightIndex];
    float currentDepth = length(fragToLight);
    float farPlane = lightFarPlanes[lightIndex];
    
    float closestDepth;
    if (lightIndex == 0) closestDepth = texture(shadowMaps[0], fragToLight).r;
    else if (lightIndex == 1) closestDepth = texture(shadowMaps[1], fragToLight).r;
    else if (lightIndex == 2) closestDepth = texture(shadowMaps[2], fragToLight).r;
    else if (lightIndex == 3) closestDepth = texture(shadowMaps[3], fragToLight).r;
    else if (lightIndex == 4) closestDepth = texture(shadowMaps[4], fragToLight).r;
    else if (lightIndex == 5) closestDepth = texture(shadowMaps[5], fragToLight).r;
    else if (lightIndex == 6) closestDepth = texture(shadowMaps[6], fragToLight).r;
    else closestDepth = texture(shadowMaps[7], fragToLight).r;
    
    closestDepth *= farPlane;
    
    // Soft bias for volumes
    float bias = 0.5;
    
    return (currentDepth - bias > closestDepth) ? 0.0 : 1.0;
}

// ============================================================================
// VOLUMETRIC SELF-SHADOWING (light marching through smoke)
// ============================================================================
float marchLightShadow(vec3 samplePos, vec3 lightDir, float lightDist, int shadowSteps) {
    if (!enableVolumeShadows) return 1.0;
    
    float shadowStepSize = min(lightDist, 10.0) / float(shadowSteps);
    float shadowDensity = 0.0;
    vec3 pos = samplePos;
    
    for (int i = 0; i < shadowSteps; i++) {
        pos += lightDir * shadowStepSize;
        
        // Check if still in volume bounds
        vec3 uvw = (pos - gridMin) / (gridMax - gridMin);
        if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) {
            break;
        }
        
        // Check for obstacle (solid geometry blocks all light)
        if (texture(obstacleTex, uvw).r > 0.5) {
            return 0.0;
        }
        
        // Accumulate density along light ray
        shadowDensity += texture(densityTex, uvw).r * shadowStepSize * shadowDensityScale;
    }
    
    // Exponential falloff based on accumulated density
    return exp(-shadowDensity);
}

// ============================================================================
// DISTANCE TO VOLUME BOUNDARY
// ============================================================================
float distToBoundary(vec3 pos, vec3 dir) {
    vec3 invDir = 1.0 / (dir + vec3(0.0001));
    vec3 t0 = (gridMin - pos) * invDir;
    vec3 t1 = (gridMax - pos) * invDir;
    vec3 tmax = max(t0, t1);
    return max(0.0, min(min(tmax.x, tmax.y), tmax.z));
}

// ============================================================================
// HEAT COLOR (for temperature visualization)
// ============================================================================
vec3 heatColor(float t) {
    return 5.0 * mix(mix(vec3(0.0), vec3(1.0, 0.0, 0.0), t),
               mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0), t*t), smoothstep(0.5, 1.0, t));
}

// ============================================================================
// MAIN RAY MARCHING LOOP
// ============================================================================
void main()
{
    // Reconstruct ray from screen coordinates
    vec4 near = vec4(vTexCoord * 2.0 - 1.0, -1.0, 1.0);
    vec4 far  = vec4(vTexCoord * 2.0 - 1.0,  1.0, 1.0);
    vec4 nearWorld = invViewProj * near;
    vec4 farWorld  = invViewProj * far;
    nearWorld /= nearWorld.w;
    farWorld  /= farWorld.w;

    vec3 rayOrigin = nearWorld.xyz;
    vec3 rayDir = normalize(farWorld.xyz - nearWorld.xyz);

    // Bounding box intersection
    vec3 invDir = 1.0 / rayDir;
    vec3 t0 = (gridMin - rayOrigin) * invDir;
    vec3 t1 = (gridMax - rayOrigin) * invDir;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tenter = max(max(tmin.x, tmin.y), tmin.z);
    float texit  = min(min(tmax.x, tmax.y), tmax.z);

    if (tenter >= texit || texit < 0.0) {
        discard;
    }

    tenter = max(tenter, 0.0);
    float t = tenter;
    vec3 pos = rayOrigin + rayDir * t;

    vec3 accumulatedColor = vec3(0.0);
    float transmittance = 1.0;

    for (int i = 0; i < maxSteps && t < texit; ++i)
    {
        vec3 uvw = (pos - gridMin) / (gridMax - gridMin);

        // Check for obstacle
        float obs = texture(obstacleTex, uvw).r;
        if (obs > 0.5) break;

        float density = texture(densityTex, uvw).r;
        float temp = texture(temperatureTex, uvw).r;

        float sampleValue = visualizeTemperature ?
            max(temp - temperatureThreshold, 0.0) : density;

        if (sampleValue > 0.01)
        {
            // Base color
            vec3 baseColor = visualizeTemperature ?
                heatColor(sampleValue * 0.1) : vec3(0.9, 0.9, 0.95);

            vec3 lightContrib = vec3(0.0);
            
            if (enableVolumeLighting) {
                // Ambient contribution
                lightContrib = ambientLight * baseColor;
                
                
                // ============================================================
                // POINT LIGHTS with Phase Function + Shadow Maps
                // ============================================================
                for (int p = 0; p < numPointLights && p < MAX_POINT_LIGHTS; p++) {
                    if (lightEnabled[p] == 0) continue;
                    
                    vec3 toLight = lightPositions[p] - pos;
                    float dist = length(toLight);
                    vec3 lightDir = toLight / dist;
                    
                    // Attenuation -- fake the distance using a custom attenuation
                    // This makes smoke highlights noticeable
                    float attenuation = lightIntensities[p] / (pow(dist, 1.7) + 1.0);
                    
                    // Shadow from solid geometry (shadow cubemaps)
                    float solidShadow = sampleVolumeShadow(p, pos);
                    
                    // Volumetric self-shadowing (light marching through smoke)
                    //float volumeShadow = marchLightShadow(pos, lightDir, dist, shadowSamples);
                    float maxMarchDist = min(dist, distToBoundary(pos, lightDir));
float volumeShadow = marchLightShadow(pos, lightDir, maxMarchDist, shadowSamples);

                    // Combined shadow factor
                    float shadow = solidShadow * volumeShadow;
                    
                    // Phase function for scattering highlights
                    float phase = phaseHG(dot(-rayDir, lightDir), phaseG);
                    
                    lightContrib += lightColors[p] * attenuation * shadow * 
                                   phase * volumeScattering * baseColor;
                
                }
            } else {
                // No lighting - just use base color
                lightContrib = baseColor;
            }

            // Beer-Lambert absorption
            float absorption = volumeAbsorption * sampleValue * stepSize;
            float sampleTrans = exp(-absorption);
            
            // Accumulate with front-to-back compositing
            accumulatedColor += transmittance * (1.0 - sampleTrans) * lightContrib;
            transmittance *= sampleTrans;

            // Early termination when nearly opaque
            if (transmittance < 0.01) break;
        }

        t += stepSize;
        pos = rayOrigin + rayDir * t;
    }

    // Blend with background
    vec3 backgroundColor = vec3(0.1, 0.15, 0.2);
    accumulatedColor += transmittance * backgroundColor;
    
    FragColor = vec4(accumulatedColor, 1.0 - transmittance);
}
)";









const char* commonVertexShaderSource = R"(
#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragColor;
out vec3 fragNormal;
out vec3 fragWorldPos;

void main() {
    vec4 worldPos = model * vec4(position, 1.0);
    fragWorldPos = worldPos.xyz;
    fragColor = color;
    fragNormal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * worldPos;
}
)";






// Common fragment shader for all primitives with point light shadows
const char* commonFragmentShaderSource = R"(
#version 430 core

in vec3 fragColor;
in vec3 fragNormal;
in vec3 fragWorldPos;

out vec4 finalColor;

// Maximum number of point lights (adjust as needed)
#define MAX_POINT_LIGHTS 8

// Point light data
uniform int numPointLights;
uniform vec3 lightPositions[MAX_POINT_LIGHTS];
uniform float lightIntensities[MAX_POINT_LIGHTS];
uniform vec3 lightColors[MAX_POINT_LIGHTS];
uniform float lightFarPlanes[MAX_POINT_LIGHTS];
uniform int lightEnabled[MAX_POINT_LIGHTS];  // NEW: 1 = enabled, 0 = disabled

// Shadow cubemaps
uniform samplerCube shadowMaps[MAX_POINT_LIGHTS];

// Camera position for specular
uniform vec3 viewPos;

// Ambient light
uniform vec3 ambientColor;
uniform float ambientStrength;

// Shadow bias to prevent acne
const float SHADOW_BIAS = 0.0125;
const float MAX_SHADOW_BIAS = 0.25;

// Sample shadow cubemap with PCF (Percentage Closer Filtering)
float calculateShadow(int lightIndex, vec3 fragToLight, float currentDepth, float farPlane) {
    // Sample directions for PCF
    vec3 sampleOffsetDirections[20] = vec3[](
        vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1),
        vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
        vec3( 1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1,  1,  0),
        vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
        vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0,  1, -1)
    );
    
    float shadow = 0.0;
    float samples = 20.0;
    float diskRadius = 0.02;
    
    // Dynamic bias based on surface angle
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(-fragToLight);
    float bias = max(MAX_SHADOW_BIAS * (1.0 - dot(normal, lightDir)), SHADOW_BIAS);
    
    for (int i = 0; i < 20; ++i) {
        float closestDepth;
        
        // Sample the appropriate shadow map based on light index
        if (lightIndex == 0) closestDepth = texture(shadowMaps[0], fragToLight + sampleOffsetDirections[i] * diskRadius).r;
        else if (lightIndex == 1) closestDepth = texture(shadowMaps[1], fragToLight + sampleOffsetDirections[i] * diskRadius).r;
        else if (lightIndex == 2) closestDepth = texture(shadowMaps[2], fragToLight + sampleOffsetDirections[i] * diskRadius).r;
        else if (lightIndex == 3) closestDepth = texture(shadowMaps[3], fragToLight + sampleOffsetDirections[i] * diskRadius).r;
        else if (lightIndex == 4) closestDepth = texture(shadowMaps[4], fragToLight + sampleOffsetDirections[i] * diskRadius).r;
        else if (lightIndex == 5) closestDepth = texture(shadowMaps[5], fragToLight + sampleOffsetDirections[i] * diskRadius).r;
        else if (lightIndex == 6) closestDepth = texture(shadowMaps[6], fragToLight + sampleOffsetDirections[i] * diskRadius).r;
        else closestDepth = texture(shadowMaps[7], fragToLight + sampleOffsetDirections[i] * diskRadius).r;
        
        closestDepth *= farPlane;
        
        if (currentDepth - bias > closestDepth) {
            shadow += 1.0;
        }
    }
    shadow /= samples;
    
    return shadow;
}

void main() {


    vec3 normal = normalize(fragNormal);


//finalColor = vec4(normal, 1.0);
//return;


    vec3 viewDir = normalize(viewPos - fragWorldPos);
    
    // Start with ambient light
    vec3 result = ambientColor * ambientStrength * fragColor;
    
// Process each point light
for (int i = 0; i < numPointLights && i < MAX_POINT_LIGHTS; ++i) {
    // Skip disabled lights
    if (lightEnabled[i] == 0) continue;
    
    vec3 lightDir = lightPositions[i] - fragWorldPos;

        float distance = length(lightDir);
        lightDir = normalize(lightDir);
        
        // Attenuation (inverse square law with intensity)
        float attenuation = lightIntensities[i] / (distance * distance);
        
        // Diffuse
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = diff * lightColors[i] * attenuation;
        
        // Specular (Blinn-Phong)
        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
        vec3 specular = spec * lightColors[i] * attenuation * 0.5;
        
        // Shadow calculation
        vec3 fragToLight = fragWorldPos - lightPositions[i];
        float currentDepth = length(fragToLight);
        float shadow = calculateShadow(i, fragToLight, currentDepth, lightFarPlanes[i]);
        
        // Apply shadow (1.0 = fully lit, 0.0 = fully shadowed)
        result += (1.0 - shadow) * (diffuse + specular) * fragColor;
    }
    
    finalColor = vec4(result, 1.0);
}
)";





// Shadow map depth shader for point lights (renders to cubemap)
const char* shadowMapVertexShader = R"(
#version 430 core
layout(location = 0) in vec3 position;

uniform mat4 model;

void main() {
    gl_Position = model * vec4(position, 1.0);
}
)";

const char* shadowMapGeometryShader = R"(
#version 430 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 18) out;

uniform mat4 shadowMatrices[6];

out vec4 FragPos;

void main() {
    for (int face = 0; face < 6; ++face) {
        gl_Layer = face;
        for (int i = 0; i < 3; ++i) {
            FragPos = gl_in[i].gl_Position;
            gl_Position = shadowMatrices[face] * FragPos;
            EmitVertex();
        }
        EndPrimitive();
    }
}
)";

const char* shadowMapFragmentShader = R"(
#version 430 core
in vec4 FragPos;

uniform vec3 lightPos;
uniform float farPlane;

void main() {
    // Get distance between fragment and light source
    float lightDistance = length(FragPos.xyz - lightPos);
    
    // Map to [0, 1] range by dividing by far plane
    lightDistance = lightDistance / farPlane;
    
    // Write as depth
    gl_FragDepth = lightDistance;
}
)";










// ============================================================================
// Shader Compilation Helper
// ============================================================================

GLuint compileComputeShader(const char* source) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        cerr << "Compute shader compilation failed:\n" << infoLog << endl;
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        cerr << "Compute program linking failed:\n" << infoLog << endl;
        return 0;
    }

    glDeleteShader(shader);
    return program;
}






// ============================================================================
// Shadow Map Shader Compilation
// ============================================================================

GLuint createShadowMapProgram() {
    GLuint vertShader = compileShader(GL_VERTEX_SHADER, shadowMapVertexShader);
    GLuint geomShader = compileShader(GL_GEOMETRY_SHADER, shadowMapGeometryShader);
    GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, shadowMapFragmentShader);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertShader);
    glAttachShader(program, geomShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        cerr << "Shadow map program linking failed:\n" << infoLog << endl;
        return 0;
    }

    glDeleteShader(vertShader);
    glDeleteShader(geomShader);
    glDeleteShader(fragShader);

    return program;
}

// ============================================================================
// Point Light Management
// ============================================================================

void addPointLight(const glm::vec3& pos, float intensity, const glm::vec3& color) {

    if (pointLights.size() >= MAX_POINT_LIGHTS) {
        cerr << "Warning: Maximum point lights (" << MAX_POINT_LIGHTS
            << ") reached, cannot add more." << endl;
        return;
    }

    PointLight light;
    light.position = pos;
    light.intensity = intensity;
    light.color = color;
    light.nearPlane = 0.1f;
    light.farPlane = 100.0f;

    // Create depth cubemap
    glGenTextures(1, &light.depthCubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, light.depthCubemap);
    for (int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT,
            SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    // Create framebuffer
    glGenFramebuffers(1, &light.shadowFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, light.shadowFBO);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, light.depthCubemap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        cerr << "Shadow map framebuffer not complete!" << endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    pointLights.push_back(light);
    cout << "Added point light at (" << pos.x << ", " << pos.y << ", " << pos.z
        << ") with intensity " << intensity << endl;
}

void initShadowMaps() {
    cout << "Initializing shadow maps..." << endl;

    // Compile shadow map shader program
    shadowMapProgram = createShadowMapProgram();
    if (!shadowMapProgram) {
        cerr << "Failed to create shadow map program!" << endl;
        return;
    }


    cout << "Shadow maps initialized with " << pointLights.size() << " point light(s)" << endl;
}

void renderShadowMaps() {
    if (!shadowMapProgram || pointLights.empty()) return;

    glUseProgram(shadowMapProgram);
    glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);

    // For each point light, render the scene to its shadow cubemap
    for (size_t lightIdx = 0; lightIdx < pointLights.size(); ++lightIdx) {
        PointLight& light = pointLights[lightIdx];

        // Skip disabled lights - no need to render their shadow maps
        if (!light.enabled) continue;

        glBindFramebuffer(GL_FRAMEBUFFER, light.shadowFBO);
        glClear(GL_DEPTH_BUFFER_BIT);

        // Create shadow projection matrix (90 degree FOV for cubemap)
        float aspect = 1.0f;
        glm::mat4 shadowProj = glm::perspective(glm::radians(90.0f), aspect,
            light.nearPlane, light.farPlane);

        // Create view matrices for each cubemap face
        glm::mat4 shadowTransforms[6];
        shadowTransforms[0] = shadowProj * glm::lookAt(light.position,
            light.position + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        shadowTransforms[1] = shadowProj * glm::lookAt(light.position,
            light.position + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        shadowTransforms[2] = shadowProj * glm::lookAt(light.position,
            light.position + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        shadowTransforms[3] = shadowProj * glm::lookAt(light.position,
            light.position + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
        shadowTransforms[4] = shadowProj * glm::lookAt(light.position,
            light.position + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        shadowTransforms[5] = shadowProj * glm::lookAt(light.position,
            light.position + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f));

        // Set uniforms
        for (int i = 0; i < 6; ++i) {
            glUniformMatrix4fv(glGetUniformLocation(shadowMapProgram,
                ("shadowMatrices[" + std::to_string(i) + "]").c_str()),
                1, GL_FALSE, glm::value_ptr(shadowTransforms[i]));
        }
        glUniform3fv(glGetUniformLocation(shadowMapProgram, "lightPos"), 1, glm::value_ptr(light.position));
        glUniform1f(glGetUniformLocation(shadowMapProgram, "farPlane"), light.farPlane);

        // Render voxel objects (shadow casters) - NOT marching cubes
        glm::mat4 identity(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shadowMapProgram, "model"), 1, GL_FALSE, glm::value_ptr(identity));

        // Draw triangles (voxel mesh) - these cast shadows
        if (numTriangleIndices > 0) {
            glBindVertexArray(triangleVAO);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(numTriangleIndices), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Restore viewport
    glViewport(0, 0, win_x, win_y);
}

void cleanupShadowMaps() {
    if (shadowMapProgram) {
        glDeleteProgram(shadowMapProgram);
        shadowMapProgram = 0;
    }

    //for (auto& light : pointLights) {
    //    if (light.shadowFBO) glDeleteFramebuffers(1, &light.shadowFBO);
    //    if (light.depthCubemap) glDeleteTextures(1, &light.depthCubemap);
    //}
    //pointLights.clear();
}










// ============================================================================
// GPU Initialization
// ============================================================================

void initGPUBuffers(std::vector<voxel_object>& objects) {
    // Compile compute shaders
    computeProgram = compileComputeShader(backgroundPointsComputeShader);
    surfaceComputeProgram = compileComputeShader(surfaceDetectionComputeShader);

    if (computeProgram == 0 || surfaceComputeProgram == 0) {
        cerr << "Failed to compile compute shaders!" << endl;
        return;
    }

    // Create render program once
    renderProgram = createShaderProgram(commonVertexShaderSource, commonFragmentShaderSource);

    // Initialize GPU data for each voxel object
    voxelObjectGPUData.resize(objects.size());

    for (size_t objIdx = 0; objIdx < objects.size(); objIdx++) {
        voxel_object& v = objects[objIdx];
        VoxelObjectGPUData& gpuData = voxelObjectGPUData[objIdx];

        // Prepare voxel centres as vec4 array
        size_t numVoxels = v.voxel_centres.size();
        vector<glm::vec4> voxelCentresVec4(numVoxels);
        for (size_t i = 0; i < numVoxels; i++) {
            voxelCentresVec4[i] = glm::vec4(v.voxel_centres[i].x, v.voxel_centres[i].y, v.voxel_centres[i].z, 0.0f);
        }

        // Create SSBOs for voxel data
        glGenBuffers(1, &gpuData.voxelCentresSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gpuData.voxelCentresSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, numVoxels * sizeof(glm::vec4), voxelCentresVec4.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &gpuData.voxelDensitiesSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gpuData.voxelDensitiesSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, numVoxels * sizeof(float), v.voxel_densities.data(), GL_STATIC_DRAW);

        // Grid min/max buffer
        struct GridMinMax {
            glm::vec4 gridMin;
            glm::vec4 gridMax;
            glm::ivec4 voxelRes;
        } gridData;

        gridData.gridMin = glm::vec4(v.vo_grid_min.x, v.vo_grid_min.y, v.vo_grid_min.z, v.cell_size);
        gridData.gridMax = glm::vec4(v.vo_grid_max.x, v.vo_grid_max.y, v.vo_grid_max.z, 0.0f);
        gridData.voxelRes = glm::ivec4(v.voxel_x_res, v.voxel_y_res, v.voxel_z_res, 0);

        glGenBuffers(1, &gpuData.gridMinMaxSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gpuData.gridMinMaxSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GridMinMax), &gridData, GL_STATIC_DRAW);

        // Create vo_grid_cells SSBO (maps cell position to voxel index)
        vector<int> gridCellsInt(v.vo_grid_cells.size());
        for (size_t i = 0; i < v.vo_grid_cells.size(); i++) {
            gridCellsInt[i] = static_cast<int>(v.vo_grid_cells[i]);
        }

        glGenBuffers(1, &gpuData.voGridCellsSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, gpuData.voGridCellsSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, gridCellsInt.size() * sizeof(int), gridCellsInt.data(), GL_STATIC_DRAW);

        cout << "Initialized GPU buffers for voxel object " << objIdx << endl;
    }

    // Create output SSBOs for background grid (shared across all objects)
    size_t gridSize = x_res * y_res * z_res;

    glGenBuffers(1, &backgroundDensitiesSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, backgroundDensitiesSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * sizeof(float), nullptr, GL_DYNAMIC_COPY);

    glGenBuffers(1, &backgroundCollisionsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, backgroundCollisionsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * sizeof(int), nullptr, GL_DYNAMIC_COPY);

    glGenBuffers(1, &surfaceDensitiesSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, surfaceDensitiesSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * sizeof(float), nullptr, GL_DYNAMIC_COPY);

    // Create persistent VAOs for rendering
    glGenVertexArrays(1, &triangleVAO);
    glGenBuffers(1, &triangleVBO);
    glGenBuffers(1, &triangleEBO);

    glGenVertexArrays(1, &pointVAO);
    glGenBuffers(1, &pointVBO);

    glGenVertexArrays(1, &axisVAO);
    glGenBuffers(1, &axisVBO);

    // Set up axis VAO (static data)
    std::vector<RenderVertex> axisVertices = {
        {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}, {{10.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, {{0.0f, 10.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}, {{0.0f, 0.0f, 10.0f}, {0.0f, 0.0f, 1.0f}},
        {{0.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}}, {{-10.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}},
        {{0.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}}, {{0.0f, -10.0f, 0.0f}, {0.5f, 0.5f, 0.5f}},
        {{0.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}}, {{0.0f, 0.0f, -10.0f}, {0.5f, 0.5f, 0.5f}}
    };

    glBindVertexArray(axisVAO);
    glBindBuffer(GL_ARRAY_BUFFER, axisVBO);
    glBufferData(GL_ARRAY_BUFFER, axisVertices.size() * sizeof(RenderVertex), axisVertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    gpuInitialized = true;
    cout << "GPU buffers initialized successfully for " << objects.size() << " voxel objects" << endl;
}



// ============================================================================
// FLUID SIMULATION - Initialization (MODIFIED for temperature)
// ============================================================================

void initFluidSimulation() {
    if (fluidInitialized) return;

    cout << "Initializing fluid simulation with temperature support..." << endl;

    // Compile fluid compute shaders
    advectionProgram = compileComputeShader(advectionComputeShader);
    GLuint velocityAdvectionProg = compileComputeShader(velocityAdvectionComputeShader);
    diffusionProgram = compileComputeShader(diffusionComputeShader);
    divergenceProgram = compileComputeShader(divergenceComputeShader);
    pressureProgram = compileComputeShader(pressureComputeShader);
    gradientSubtractProgram = compileComputeShader(gradientSubtractComputeShader);
    boundaryProgram = compileComputeShader(boundaryComputeShader);
    addSourceProgram = compileComputeShader(addSourceComputeShader);
    turbulenceProgram = compileComputeShader(turbulenceComputeShader);
    obstacleProgram = compileComputeShader(obstacleComputeShader);

    // NEW: Compile buoyancy program
    buoyancyProgram = compileComputeShader(buoyancyComputeShader);


    volumeRenderProgram = createShaderProgram(volumeVertSource, volumeFragSource);
    initFullscreenQuad();


    initMarchingCubes();


    // Store velocity advection program (we'll use it separately)
    static GLuint velAdvProg = velocityAdvectionProg;

    if (!advectionProgram || !diffusionProgram || !divergenceProgram ||
        !pressureProgram || !gradientSubtractProgram || !boundaryProgram ||
        !addSourceProgram || !turbulenceProgram || !obstacleProgram || !buoyancyProgram) {
        cerr << "Failed to compile fluid simulation shaders!" << endl;
        return;
    }

    size_t gridSize = x_res * y_res * z_res;

    // Initialize velocity buffers (double buffered)
    vector<glm::vec4> zeroVelocity(gridSize, glm::vec4(0.0f));
    for (int i = 0; i < 2; i++) {
        glGenBuffers(1, &velocitySSBO[i]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, velocitySSBO[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * sizeof(glm::vec4), zeroVelocity.data(), GL_DYNAMIC_COPY);
    }

    // Initialize density buffers (double buffered)
    vector<float> zeroDensity(gridSize, 0.0f);
    for (int i = 0; i < 2; i++) {
        glGenBuffers(1, &densitySSBO[i]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, densitySSBO[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * sizeof(float), zeroDensity.data(), GL_DYNAMIC_COPY);
    }

    // Initialize pressure buffers (double buffered)
    for (int i = 0; i < 2; i++) {
        glGenBuffers(1, &pressureSSBO[i]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, pressureSSBO[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * sizeof(float), zeroDensity.data(), GL_DYNAMIC_COPY);
    }

    // Initialize divergence buffer
    glGenBuffers(1, &divergenceSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, divergenceSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * sizeof(float), zeroDensity.data(), GL_DYNAMIC_COPY);

    // Initialize obstacle buffer
    glGenBuffers(1, &obstacleSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, obstacleSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * sizeof(float), zeroDensity.data(), GL_DYNAMIC_COPY);

    // Initialize turbulent viscosity buffer
    glGenBuffers(1, &turbulentViscositySSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, turbulentViscositySSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * sizeof(float), zeroDensity.data(), GL_DYNAMIC_COPY);

    // ========================================
    // NEW: Initialize temperature buffers
    // ========================================
    vector<float> ambientTemp(gridSize, fluidParams.ambientTemperature);
    for (int i = 0; i < 2; i++) {
        glGenBuffers(1, &temperatureSSBO[i]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, temperatureSSBO[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * sizeof(float), ambientTemp.data(), GL_DYNAMIC_COPY);
    }
    cout << "Temperature buffers initialized to ambient: " << fluidParams.ambientTemperature << endl;

    // Create fluid visualization VAO
    glGenVertexArrays(1, &fluidVAO);
    glGenBuffers(1, &fluidVBO);

    fluidInitialized = true;
    cout << "Fluid simulation initialized successfully with:" << endl;
    cout << "  - Buoyancy: " << (fluidParams.enableBuoyancy ? "ON" : "OFF") << endl;
    cout << "  - Gravity: " << (fluidParams.enableGravity ? "ON" : "OFF") << endl;
    cout << "  - Temperature: " << (fluidParams.enableTemperature ? "ON" : "OFF") << endl;
}

// ============================================================================
// FLUID SIMULATION - Update Obstacles from Voxels
// ============================================================================

void updateFluidObstacles() {
    if (!fluidInitialized || !gpuInitialized) return;

    glUseProgram(obstacleProgram);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, backgroundDensitiesSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, obstacleSSBO);

    glUniform3i(glGetUniformLocation(obstacleProgram, "gridRes"), x_res, y_res, z_res);

    GLuint groupsX = (x_res + 7) / 8;
    GLuint groupsY = (y_res + 7) / 8;
    GLuint groupsZ = (z_res + 7) / 8;
    glDispatchCompute(groupsX, groupsY, groupsZ);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ============================================================================
// FLUID SIMULATION - Main Update Step (MODIFIED with buoyancy)
// ============================================================================

void stepFluidSimulation() {
    if (!fluidInitialized || !fluidSimEnabled) return;

    GLuint groupsX = (x_res + 7) / 8;
    GLuint groupsY = (y_res + 7) / 8;
    GLuint groupsZ = (z_res + 7) / 8;

    glm::vec3 bgGridMin(0, 0, 0);
    glm::vec3 bgGridMax(x_grid_max, y_grid_max, z_grid_max);

    int src = currentBuffer;
    int dst = 1 - currentBuffer;

    // ========================================
    // 1. Apply external forces (GRAVITY + BUOYANCY) - NEW!
    // ========================================
    if (fluidParams.enableGravity || fluidParams.enableBuoyancy) {
        glUseProgram(buoyancyProgram);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, velocitySSBO[src]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, densitySSBO[0]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, temperatureSSBO[0]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, obstacleSSBO);

        glUniform3i(glGetUniformLocation(buoyancyProgram, "gridRes"), x_res, y_res, z_res);
        glUniform1f(glGetUniformLocation(buoyancyProgram, "dt"), fluidParams.dt);
        glUniform1f(glGetUniformLocation(buoyancyProgram, "ambientTemperature"), fluidParams.ambientTemperature);
        glUniform1f(glGetUniformLocation(buoyancyProgram, "buoyancyAlpha"), fluidParams.buoyancyAlpha);
        glUniform1f(glGetUniformLocation(buoyancyProgram, "buoyancyBeta"), fluidParams.buoyancyBeta);
        glUniform1f(glGetUniformLocation(buoyancyProgram, "gravity"), fluidParams.gravity);
        glUniform3fv(glGetUniformLocation(buoyancyProgram, "gravityDirection"), 1,
            glm::value_ptr(fluidParams.gravityDirection));
        glUniform1i(glGetUniformLocation(buoyancyProgram, "enableGravity"), fluidParams.enableGravity ? 1 : 0);
        glUniform1i(glGetUniformLocation(buoyancyProgram, "enableBuoyancy"), fluidParams.enableBuoyancy ? 1 : 0);

        glDispatchCompute(groupsX, groupsY, groupsZ);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }


    // ========================================
    // 3. Advect velocity (semi-Lagrangian)
    // ========================================
    static GLuint velAdvProg = 0;
    if (velAdvProg == 0) {
        velAdvProg = compileComputeShader(velocityAdvectionComputeShader);
    }

    glUseProgram(velAdvProg);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, velocitySSBO[src]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, velocitySSBO[dst]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, obstacleSSBO);

    glUniform3i(glGetUniformLocation(velAdvProg, "gridRes"), x_res, y_res, z_res);
    glUniform3fv(glGetUniformLocation(velAdvProg, "gridMin"), 1, glm::value_ptr(bgGridMin));
    glUniform3fv(glGetUniformLocation(velAdvProg, "gridMax"), 1, glm::value_ptr(bgGridMax));
    glUniform1f(glGetUniformLocation(velAdvProg, "dt"), fluidParams.dt);
    glUniform1f(glGetUniformLocation(velAdvProg, "dissipation"), fluidParams.velocityDissipation);

    glDispatchCompute(groupsX, groupsY, groupsZ);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    std::swap(src, dst);

    // ========================================
    // 4. Advect density
    // ========================================
    glUseProgram(advectionProgram);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, velocitySSBO[src]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, densitySSBO[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, densitySSBO[1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, obstacleSSBO);

    glUniform3i(glGetUniformLocation(advectionProgram, "gridRes"), x_res, y_res, z_res);
    glUniform3fv(glGetUniformLocation(advectionProgram, "gridMin"), 1, glm::value_ptr(bgGridMin));
    glUniform3fv(glGetUniformLocation(advectionProgram, "gridMax"), 1, glm::value_ptr(bgGridMax));
    glUniform1f(glGetUniformLocation(advectionProgram, "dt"), fluidParams.dt);
    glUniform1f(glGetUniformLocation(advectionProgram, "dissipation"), fluidParams.densityDissipation);

    glDispatchCompute(groupsX, groupsY, groupsZ);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Swap density buffers
    std::swap(densitySSBO[0], densitySSBO[1]);

    // ========================================
    // 5. Advect temperature - NEW!
    // ========================================
    if (fluidParams.enableTemperature) {
        glUseProgram(advectionProgram);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, velocitySSBO[src]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, temperatureSSBO[0]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, temperatureSSBO[1]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, obstacleSSBO);

        glUniform3i(glGetUniformLocation(advectionProgram, "gridRes"), x_res, y_res, z_res);
        glUniform3fv(glGetUniformLocation(advectionProgram, "gridMin"), 1, glm::value_ptr(bgGridMin));
        glUniform3fv(glGetUniformLocation(advectionProgram, "gridMax"), 1, glm::value_ptr(bgGridMax));
        glUniform1f(glGetUniformLocation(advectionProgram, "dt"), fluidParams.dt);
        glUniform1f(glGetUniformLocation(advectionProgram, "dissipation"), fluidParams.temperatureDissipation);

        glDispatchCompute(groupsX, groupsY, groupsZ);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Swap temperature buffers
        std::swap(temperatureSSBO[0], temperatureSSBO[1]);
    }

    // ========================================
    // 6. Compute divergence
    // ========================================
    glUseProgram(divergenceProgram);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, velocitySSBO[src]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, divergenceSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, obstacleSSBO);

    glUniform3i(glGetUniformLocation(divergenceProgram, "gridRes"), x_res, y_res, z_res);
    glUniform3fv(glGetUniformLocation(divergenceProgram, "gridMin"), 1, glm::value_ptr(bgGridMin));
    glUniform3fv(glGetUniformLocation(divergenceProgram, "gridMax"), 1, glm::value_ptr(bgGridMax));

    glDispatchCompute(groupsX, groupsY, groupsZ);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // ========================================
    // 7. Solve pressure (Jacobi iteration)
    // ========================================
    // Clear pressure
    size_t gridSize = x_res * y_res * z_res;
    vector<float> zeroPressure(gridSize, 0.0f);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pressureSSBO[0]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * sizeof(float), zeroPressure.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pressureSSBO[1]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * sizeof(float), zeroPressure.data());

    glUseProgram(pressureProgram);
    glUniform3i(glGetUniformLocation(pressureProgram, "gridRes"), x_res, y_res, z_res);
    glUniform3fv(glGetUniformLocation(pressureProgram, "gridMin"), 1, glm::value_ptr(bgGridMin));
    glUniform3fv(glGetUniformLocation(pressureProgram, "gridMax"), 1, glm::value_ptr(bgGridMax));

    int pSrc = 0;
    int pDst = 1;

    for (int i = 0; i < fluidParams.jacobiIterations; i++) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pressureSSBO[pSrc]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pressureSSBO[pDst]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, divergenceSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, obstacleSSBO);

        glDispatchCompute(groupsX, groupsY, groupsZ);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        std::swap(pSrc, pDst);
    }

    // ========================================
    // 8. Subtract pressure gradient
    // ========================================
    glUseProgram(gradientSubtractProgram);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, velocitySSBO[src]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pressureSSBO[pSrc]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, obstacleSSBO);

    glUniform3i(glGetUniformLocation(gradientSubtractProgram, "gridRes"), x_res, y_res, z_res);
    glUniform3fv(glGetUniformLocation(gradientSubtractProgram, "gridMin"), 1, glm::value_ptr(bgGridMin));
    glUniform3fv(glGetUniformLocation(gradientSubtractProgram, "gridMax"), 1, glm::value_ptr(bgGridMax));

    glDispatchCompute(groupsX, groupsY, groupsZ);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // ========================================
    // 9. Apply boundary conditions
    // ========================================
    glUseProgram(boundaryProgram);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, velocitySSBO[src]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, densitySSBO[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, obstacleSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, temperatureSSBO[0]);

    glUniform3i(glGetUniformLocation(boundaryProgram, "gridRes"), x_res, y_res, z_res);
    glUniform1f(glGetUniformLocation(boundaryProgram, "ambientTemperature"), fluidParams.ambientTemperature);

    glDispatchCompute(groupsX, groupsY, groupsZ);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Update current buffer index
    currentBuffer = src;
}

// ============================================================================
// FLUID SIMULATION - Add Source (MODIFIED with temperature)
// ============================================================================

void addFluidSource(const glm::vec3& position, const glm::vec3& velocity, bool addDens, bool addVel) {
    if (!fluidInitialized) return;

    glUseProgram(addSourceProgram);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, densitySSBO[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, velocitySSBO[currentBuffer]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, obstacleSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, temperatureSSBO[0]);

    glm::vec3 bgGridMin(0,0, 0);
    glm::vec3 bgGridMax(x_grid_max, y_grid_max, z_grid_max);

    glUniform3i(glGetUniformLocation(addSourceProgram, "gridRes"), x_res, y_res, z_res);
    glUniform3fv(glGetUniformLocation(addSourceProgram, "gridMin"), 1, glm::value_ptr(bgGridMin));
    glUniform3fv(glGetUniformLocation(addSourceProgram, "gridMax"), 1, glm::value_ptr(bgGridMax));
    glUniform3fv(glGetUniformLocation(addSourceProgram, "sourcePos"), 1, glm::value_ptr(position));
    glUniform3fv(glGetUniformLocation(addSourceProgram, "sourceVelocity"), 1, glm::value_ptr(velocity));
    glUniform1f(glGetUniformLocation(addSourceProgram, "sourceRadius"),
        (x_grid_max * 2.0f / x_res) * injectRadius);
    glUniform1f(glGetUniformLocation(addSourceProgram, "densityAmount"), fluidParams.densityAmount);
    glUniform1f(glGetUniformLocation(addSourceProgram, "velocityAmount"), fluidParams.velocityAmount);
    glUniform1f(glGetUniformLocation(addSourceProgram, "temperatureAmount"), fluidParams.temperatureAmount);
    glUniform1i(glGetUniformLocation(addSourceProgram, "addDensity"), addDens ? 1 : 0);
    glUniform1i(glGetUniformLocation(addSourceProgram, "addVelocity"), addVel ? 1 : 0);
    // Add temperature when adding density (for fire/smoke effect)
    glUniform1i(glGetUniformLocation(addSourceProgram, "addTemperature"),
        (addDens && fluidParams.enableTemperature) ? 1 : 0);

    GLuint groupsX = (x_res + 7) / 8;
    GLuint groupsY = (y_res + 7) / 8;
    GLuint groupsZ = (z_res + 7) / 8;
    glDispatchCompute(groupsX, groupsY, groupsZ);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ============================================================================
// FLUID SIMULATION - Visualization (MODIFIED with temperature option)
// ============================================================================

// ============================================================================
// MARCHING CUBES - Initialization Function
// Add this to be called from initFluidSimulation() or initGPUBuffers()
// ============================================================================

void initMarchingCubes() {
    cout << "Initializing Marching Cubes..." << endl;

    // Compile compute shader
    mcComputeProgram = compileComputeShader(marchingCubesComputeShader);
    if (!mcComputeProgram) {
        cerr << "Failed to compile Marching Cubes compute shader!" << endl;
        return;
    }

    // Compile render shaders
    mcRenderProgram = createShaderProgram(mcVertexShaderSource, mcFragmentShaderSource);
    if (!mcRenderProgram) {
        cerr << "Failed to compile Marching Cubes render shaders!" << endl;
        return;
    }

    // Create edge table SSBO
    glGenBuffers(1, &mcEdgeTableSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mcEdgeTableSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 256 * sizeof(int), edgeTable, GL_STATIC_DRAW);

    // Create triangle table SSBO (flattened 256 * 16)
    vector<int> flatTriTable(256 * 16);
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 16; j++) {
            flatTriTable[i * 16 + j] = triTable[i][j];
        }
    }
    glGenBuffers(1, &mcTriTableSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mcTriTableSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, flatTriTable.size() * sizeof(int), flatTriTable.data(), GL_STATIC_DRAW);

    // Create vertex counter SSBO
    glGenBuffers(1, &mcVertexCountSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mcVertexCountSSBO);
    GLuint zero = 0;
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), &zero, GL_DYNAMIC_DRAW);

    // Create output vertex buffer (large enough for all layers)
    size_t totalMaxVerts = MC_MAX_VERTICES * NUM_ISO_LAYERS;
    glGenBuffers(1, &mcVertexBufferSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mcVertexBufferSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, totalMaxVerts * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);

    // Create output normal buffer
    glGenBuffers(1, &mcNormalBufferSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mcNormalBufferSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, totalMaxVerts * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);

    // Create VAO for rendering
    glGenVertexArrays(1, &mcVAO);
    glBindVertexArray(mcVAO);

    // Bind vertex buffer as VBO for rendering
    glBindBuffer(GL_ARRAY_BUFFER, mcVertexBufferSSBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
    glEnableVertexAttribArray(0);

    // Bind normal buffer
    glBindBuffer(GL_ARRAY_BUFFER, mcNormalBufferSSBO);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    cout << "Marching Cubes initialized successfully" << endl;
}

// ============================================================================
// MARCHING CUBES - Run compute shader for a single isovalue
// ============================================================================

void runMarchingCubes(float isoValue, int layerIndex) {
    if (!mcComputeProgram || !fluidInitialized) return;

    // Calculate vertex offset for this layer
    GLuint vertexOffset = 0;
    for (int i = 0; i < layerIndex; i++) {
        vertexOffset += mcLayerVertexCounts[i];
    }
    mcLayerVertexOffsets[layerIndex] = vertexOffset;

    // Reset vertex counter for this layer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mcVertexCountSSBO);
    GLuint zero = 0;
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &zero);

    glUseProgram(mcComputeProgram);

    // Bind SSBOs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, densitySSBO[currentBuffer]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mcEdgeTableSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mcTriTableSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mcVertexBufferSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, mcNormalBufferSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, mcVertexCountSSBO);

    // Set uniforms
    glUniform3i(glGetUniformLocation(mcComputeProgram, "gridRes"), x_res, y_res, z_res);
    glUniform3f(glGetUniformLocation(mcComputeProgram, "gridMin"), 0, 0, 0);
    glUniform3f(glGetUniformLocation(mcComputeProgram, "gridMax"), x_grid_max, y_grid_max, z_grid_max);
    glUniform1f(glGetUniformLocation(mcComputeProgram, "isoValue"), isoValue);
    glUniform1ui(glGetUniformLocation(mcComputeProgram, "maxVertices"), MC_MAX_VERTICES);
    glUniform1ui(glGetUniformLocation(mcComputeProgram, "vertexOffset"), vertexOffset);

    // Dispatch compute shader
    GLuint groupsX = (x_res + 3) / 4;
    GLuint groupsY = (y_res + 3) / 4;
    GLuint groupsZ = (z_res + 3) / 4;
    glDispatchCompute(groupsX, groupsY, groupsZ);

    // Wait for compute shader to finish
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

    // Read back vertex count for this layer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, mcVertexCountSSBO);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &mcLayerVertexCounts[layerIndex]);
}

// ============================================================================
// MARCHING CUBES - Run all layers
// ============================================================================

void runAllMarchingCubesLayers() {
    if (!mcComputeProgram) return;

    // Run marching cubes for each isovalue layer
    for (int i = 0; i < NUM_ISO_LAYERS; i++) {
        runMarchingCubes(isoValues[i], i);
    }
}

// ============================================================================
// MARCHING CUBES - Draw all layer meshes with transparency
// ============================================================================

void drawMarchingCubesMesh() {
    if (!mcRenderProgram || !mcVAO) return;

    // Check if any vertices were generated
    GLuint totalVerts = 0;
    for (int i = 0; i < NUM_ISO_LAYERS; i++) {
        totalVerts += mcLayerVertexCounts[i];
    }
    if (totalVerts == 0) return;

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Disable depth writing for transparent objects (but keep depth test)
    glDepthMask(GL_FALSE);

    // Disable backface culling for double-sided rendering
    glDisable(GL_CULL_FACE);

    glUseProgram(mcRenderProgram);

    // Set common uniforms
    glm::mat4 model = glm::mat4(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(mcRenderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(mcRenderProgram, "view"), 1, GL_FALSE, glm::value_ptr(main_camera.view_mat));
    glUniformMatrix4fv(glGetUniformLocation(mcRenderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(main_camera.projection_mat));
    glUniform3fv(glGetUniformLocation(mcRenderProgram, "viewPos"), 1, &main_camera.eye.x);
    glUniform3f(glGetUniformLocation(mcRenderProgram, "lightDir"), -0.5f, -1.0f, -0.3f);

    glBindVertexArray(mcVAO);

    // Draw layers from outer (most transparent) to inner (most opaque)
    // This gives better transparency compositing
    for (int i = 0; i < NUM_ISO_LAYERS; i++) {
        if (mcLayerVertexCounts[i] == 0) continue;

        // Set layer-specific color and opacity
        glm::vec4 color = isoColors[i];

        // If visualizing temperature, modify colors based on layer
        if (fluidParams.visualizeTemperature) {
            // Temperature gradient: blue (cold) -> white -> yellow -> red (hot)
            float t = float(i) / float(NUM_ISO_LAYERS - 1);
            if (t < 0.33f) {
                color = glm::mix(glm::vec4(0.2f, 0.4f, 1.0f, isoOpacities[i]),
                    glm::vec4(1.0f, 1.0f, 1.0f, isoOpacities[i]), t * 3.0f);
            }
            else if (t < 0.66f) {
                color = glm::mix(glm::vec4(1.0f, 1.0f, 1.0f, isoOpacities[i]),
                    glm::vec4(1.0f, 1.0f, 0.0f, isoOpacities[i]), (t - 0.33f) * 3.0f);
            }
            else {
                color = glm::mix(glm::vec4(1.0f, 1.0f, 0.0f, isoOpacities[i]),
                    glm::vec4(1.0f, 0.3f, 0.0f, isoOpacities[i]), (t - 0.66f) * 3.0f);
            }
        }

        glUniform4fv(glGetUniformLocation(mcRenderProgram, "layerColor"), 1, glm::value_ptr(color));

        // Draw this layer's triangles
        GLint firstVertex = mcLayerVertexOffsets[i];
        GLsizei vertexCount = mcLayerVertexCounts[i];
        glDrawArrays(GL_TRIANGLES, firstVertex, vertexCount);
    }

    glBindVertexArray(0);

    // Restore state
    glDepthMask(GL_TRUE);
    glEnable(GL_CULL_FACE);
    glDisable(GL_BLEND);
}

// ============================================================================
// MARCHING CUBES - Cleanup
// Add to your cleanup() function
// ============================================================================

void cleanupMarchingCubes() {
    if (mcComputeProgram) glDeleteProgram(mcComputeProgram);
    if (mcRenderProgram) glDeleteProgram(mcRenderProgram);

    glDeleteBuffers(1, &mcEdgeTableSSBO);
    glDeleteBuffers(1, &mcTriTableSSBO);
    glDeleteBuffers(1, &mcVertexCountSSBO);
    glDeleteBuffers(1, &mcVertexBufferSSBO);
    glDeleteBuffers(1, &mcNormalBufferSSBO);

    glDeleteVertexArrays(1, &mcVAO);
}


void updateFluidVisualization() {
    if (!fluidInitialized) return;

    size_t gridSize = x_res * y_res * z_res;

    // Read back density
    vector<float> densities(gridSize);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, densitySSBO[0]);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * sizeof(float), densities.data());

    // Read back temperature if visualizing it
    vector<float> temperatures;
    if (fluidParams.visualizeTemperature) {
        temperatures.resize(gridSize);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, temperatureSSBO[0]);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * sizeof(float), temperatures.data());
    }

    // Read back obstacles for exclusion
    vector<float> obstacles(gridSize);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, obstacleSSBO);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * sizeof(float), obstacles.data());

    // Build visualization points
    vector<RenderVertex> fluidVertices;
    fluidVertices.reserve(gridSize / 10);

    float x_grid_min = 0;
    float y_grid_min = 0;
    float z_grid_min = 0;

    float x_step = (x_grid_max - x_grid_min) / (x_res - 1);
    float y_step = (y_grid_max - y_grid_min) / (y_res - 1);
    float z_step = (z_grid_max - z_grid_min) / (z_res - 1);

    for (size_t z = 0; z < z_res; z++) {
        for (size_t y = 0; y < y_res; y++) {
            for (size_t x = 0; x < x_res; x++) {
                size_t idx = x + y * x_res + z * x_res * y_res;

                // Skip obstacles
                if (obstacles[idx] > 0.5f) {
                    continue;
                }

                // Only show cells with density above threshold
                if (densities[idx] > densityThreshold) {
                    RenderVertex rv;
                    rv.position[0] = x_grid_min + x * x_step;
                    rv.position[1] = y_grid_min + y * y_step;
                    rv.position[2] = z_grid_min + z * z_step;

                    if (fluidParams.visualizeTemperature && !temperatures.empty()) {
                        // Temperature visualization: cold (blue) -> hot (red/yellow)
                        float t = std::min(temperatures[idx] / 20.0f, 1.0f); // Normalize temperature
                        float d = std::min(densities[idx] / 50.0f, 1.0f);

                        // Hot = orange/yellow, cold = blue/cyan
                        rv.color[0] = t;                          // Red increases with temp
                        rv.color[1] = t * 0.5f;                   // Some green for yellow
                        rv.color[2] = (1.0f - t) * d;             // Blue decreases with temp
                    }
                    else {
                        // Standard density visualization (blue to red heat map)
                        float d = std::min(densities[idx] / 50.0f, 1.0f);
                        rv.color[0] = d;                    // Red
                        rv.color[1] = 0.2f * (1.0f - d);    // Green
                        rv.color[2] = 1.0f - d;             // Blue
                    }

                    fluidVertices.push_back(rv);
                }
            }
        }
    }

    numFluidPoints = fluidVertices.size();

    // Update VBO
    glBindVertexArray(fluidVAO);
    glBindBuffer(GL_ARRAY_BUFFER, fluidVBO);
    glBufferData(GL_ARRAY_BUFFER, fluidVertices.size() * sizeof(RenderVertex),
        fluidVertices.empty() ? nullptr : fluidVertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}







void draw_fluid_fast() {
    if (!fluidInitialized) return;

    // Update textures from SSBOs
    updateFluidTextures();

    if (useMarchingCubes) {
        // Run marching cubes on current density field
        runAllMarchingCubesLayers();
        drawMarchingCubesMesh();
    }
    else {
        // Ray marching with lighting
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(volumeRenderProgram);

        // Bind density texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, densityTexture);
        glUniform1i(glGetUniformLocation(volumeRenderProgram, "densityTex"), 0);

        // Bind temperature texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_3D, temperatureTexture);
        glUniform1i(glGetUniformLocation(volumeRenderProgram, "temperatureTex"), 1);

        // Bind obstacle texture
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_3D, obstacleTexture);
        glUniform1i(glGetUniformLocation(volumeRenderProgram, "obstacleTex"), 2);

        // ====================================================================
        // BIND SHADOW CUBEMAPS (starting at texture unit 3)
        // ====================================================================
        int numLights = std::min((int)pointLights.size(), 8);
        glUniform1i(glGetUniformLocation(volumeRenderProgram, "numPointLights"), numLights);

        for (int i = 0; i < numLights; ++i) {
            // Bind shadow cubemap
            glActiveTexture(GL_TEXTURE3 + i);
            glBindTexture(GL_TEXTURE_CUBE_MAP, pointLights[i].depthCubemap);
            std::string uniformName = "shadowMaps[" + std::to_string(i) + "]";
            glUniform1i(glGetUniformLocation(volumeRenderProgram, uniformName.c_str()), 3 + i);

            // Pass light position
            uniformName = "lightPositions[" + std::to_string(i) + "]";
            glUniform3fv(glGetUniformLocation(volumeRenderProgram, uniformName.c_str()),
                1, glm::value_ptr(pointLights[i].position));

            // Pass light intensity
            uniformName = "lightIntensities[" + std::to_string(i) + "]";
            glUniform1f(glGetUniformLocation(volumeRenderProgram, uniformName.c_str()),
                pointLights[i].intensity);

            // Pass light color
            uniformName = "lightColors[" + std::to_string(i) + "]";
            glUniform3fv(glGetUniformLocation(volumeRenderProgram, uniformName.c_str()),
                1, glm::value_ptr(pointLights[i].color));

            // Pass far plane
            uniformName = "lightFarPlanes[" + std::to_string(i) + "]";
            glUniform1f(glGetUniformLocation(volumeRenderProgram, uniformName.c_str()),
                pointLights[i].farPlane);

            // Pass enabled state
            uniformName = "lightEnabled[" + std::to_string(i) + "]";
            glUniform1i(glGetUniformLocation(volumeRenderProgram, uniformName.c_str()),
                pointLights[i].enabled ? 1 : 0);
        }

        // ====================================================================
        // SET ADDITIONAL LIGHT UNIFORMS (directional, spot lights)
        // ====================================================================


        // ====================================================================
        // CAMERA AND GRID UNIFORMS
        // ====================================================================
        glm::mat4 viewProj = main_camera.projection_mat * main_camera.view_mat;
        glm::mat4 invViewProj = glm::inverse(viewProj);

        glUniformMatrix4fv(glGetUniformLocation(volumeRenderProgram, "invViewProj"),
            1, GL_FALSE, &invViewProj[0][0]);
        glUniform3fv(glGetUniformLocation(volumeRenderProgram, "cameraPos"),
            1, &main_camera.eye.x);
        glUniform3f(glGetUniformLocation(volumeRenderProgram, "gridMin"),
            0, 0, 9);
        glUniform3f(glGetUniformLocation(volumeRenderProgram, "gridMax"),
            x_grid_max, y_grid_max, z_grid_max);
        glUniform3i(glGetUniformLocation(volumeRenderProgram, "gridRes"),
            x_res, y_res, z_res);
        glUniform1i(glGetUniformLocation(volumeRenderProgram, "visualizeTemperature"),
            fluidParams.visualizeTemperature ? 1 : 0);

        // Ray marching parameters
        glUniform1f(glGetUniformLocation(volumeRenderProgram, "densityFactor"), 1.0f);
        glUniform1f(glGetUniformLocation(volumeRenderProgram, "temperatureThreshold"), 1.0f);
        glUniform1f(glGetUniformLocation(volumeRenderProgram, "stepSize"), 0.15f);
        glUniform1i(glGetUniformLocation(volumeRenderProgram, "maxSteps"), 256);

        // ====================================================================
        // VOLUME LIGHTING PARAMETERS (Phase Function Scattering)
        // ====================================================================
        glUniform1f(glGetUniformLocation(volumeRenderProgram, "volumeAbsorption"), fluidParams.volumeAbsorption);
        glUniform1f(glGetUniformLocation(volumeRenderProgram, "volumeScattering"), fluidParams.volumeScattering);
        glUniform1i(glGetUniformLocation(volumeRenderProgram, "shadowSamples"), fluidParams.shadowSamples);
        glUniform1f(glGetUniformLocation(volumeRenderProgram, "shadowDensityScale"), fluidParams.shadowDensityScale);
        glUniform1f(glGetUniformLocation(volumeRenderProgram, "phaseG"), fluidParams.phaseG);
        glUniform1i(glGetUniformLocation(volumeRenderProgram, "enableVolumeShadows"), fluidParams.enableVolumeShadows ? 1 : 0);
        glUniform1i(glGetUniformLocation(volumeRenderProgram, "enableVolumeLighting"), fluidParams.enableVolumeLighting ? 1 : 0);
        glUniform3f(glGetUniformLocation(volumeRenderProgram, "ambientLight"), 0.15f, 0.15f, 0.2f);

        // Draw fullscreen quad
        glBindVertexArray(fullscreenVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        // ====================================================================
        // UNBIND SHADOW MAPS
        // ====================================================================
        for (int i = 0; i < numLights; ++i) {
            glActiveTexture(GL_TEXTURE3 + i);
            glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
        }

        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
    }
}






// ============================================================================
// Mouse Ray Casting for 3D Position
// ============================================================================







glm::vec3 screenToWorld_ortho(int mouseX, int mouseY, float depth) {
    // Convert screen coordinates to normalized device coordinates
    float ndcX = (2.0f * mouseX) / win_x - 1.0f;
    float ndcY = 1.0f - (2.0f * mouseY) / win_y;

    // For orthographic projection, we unproject two points at different depths
    // to get the ray origin and direction
    glm::mat4 invViewProj = glm::inverse(main_camera.projection_mat * main_camera.view_mat);

    // Unproject point on near plane (z = -1 in NDC)
    glm::vec4 nearPoint = invViewProj * glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
    nearPoint /= nearPoint.w;

    // Unproject point on far plane (z = 1 in NDC)
    glm::vec4 farPoint = invViewProj * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
    farPoint /= farPoint.w;

    // Ray origin and direction (for ortho, rays are parallel)
    glm::vec3 rayOrigin = glm::vec3(nearPoint);
    glm::vec3 rayDir = glm::normalize(glm::vec3(farPoint) - glm::vec3(nearPoint));

    // Try to intersect with the Y=0 plane (ground plane)
    if (std::abs(rayDir.y) > 0.001f) {
        float t = -rayOrigin.y / rayDir.y;
        if (t > 0) {
            return rayOrigin + rayDir * t;
        }
    }

    // If Y=0 intersection fails, try Z = z_grid_max/2 plane (middle of grid depth)
    float targetZ = z_grid_max / 2.0f;
    if (std::abs(rayDir.z) > 0.001f) {
        float t = (targetZ - rayOrigin.z) / rayDir.z;
        if (t > 0) {
            return rayOrigin + rayDir * t;
        }
    }

    // Fallback: use a point along the ray at a reasonable distance
    return rayOrigin + rayDir * 50.0f;
}




glm::vec3 screenToWorld(int mouseX, int mouseY, float depth) {
    // Convert screen coordinates to normalized device coordinates
    float x = (2.0f * mouseX) / win_x - 1.0f;
    float y = 1.0f - (2.0f * mouseY) / win_y;
    float z = depth;

    glm::vec4 clipCoords(x, y, z, 1.0f);

    // Inverse projection
    glm::mat4 invProj = glm::inverse(main_camera.projection_mat);
    glm::vec4 eyeCoords = invProj * clipCoords;
    eyeCoords.z = -1.0f;
    eyeCoords.w = 0.0f;

    // Inverse view
    glm::mat4 invView = glm::inverse(main_camera.view_mat);
    glm::vec4 worldRay = invView * eyeCoords;
    glm::vec3 rayDir = glm::normalize(glm::vec3(worldRay));

    // Get camera position
    glm::vec3 camPos = glm::vec3(invView[3]);

    // Intersect with Y=0 plane (or use a default depth)
    float t = -camPos.y / rayDir.y;
    if (t < 0 || std::abs(rayDir.y) < 0.001f) {
        // Use a fixed distance if no good intersection
        t = 10.0f;
    }

    return camPos + rayDir * t;
}

// ============================================================================
// GPU Background Points Computation
// ============================================================================



void get_background_points_GPU(std::vector<voxel_object>& objects) {
    if (!gpuInitialized) return;

    size_t gridSize = x_res * y_res * z_res;

    // Clear background densities to 0 before processing all objects
    vector<float> zeroDensities(gridSize, 0.0f);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, backgroundDensitiesSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * sizeof(float), zeroDensities.data());

    vector<int> negOneCollisions(gridSize, -1);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, backgroundCollisionsSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * sizeof(int), negOneCollisions.data());

    glm::vec3 bgGridMin(0, 0, 0);
    glm::vec3 bgGridMax(x_grid_max, y_grid_max, z_grid_max);

    GLuint groupsX = (x_res + 7) / 8;
    GLuint groupsY = (y_res + 7) / 8;
    GLuint groupsZ = (z_res + 7) / 8;

    // Process each voxel object - results are accumulated (union of all obstacles)
    for (size_t objIdx = 0; objIdx < objects.size(); objIdx++) {
        voxel_object& v = objects[objIdx];
        VoxelObjectGPUData& gpuData = voxelObjectGPUData[objIdx];

        glUseProgram(computeProgram);

        // Bind SSBOs for this object
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gpuData.voxelCentresSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gpuData.voxelDensitiesSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gpuData.gridMinMaxSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, backgroundDensitiesSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, backgroundCollisionsSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, gpuData.voGridCellsSSBO);

        // Set uniforms
        glm::mat4 invModel = glm::inverse(v.model_matrix);
        glUniformMatrix4fv(glGetUniformLocation(computeProgram, "invModelMatrix"), 1, GL_FALSE, glm::value_ptr(invModel));

        glUniform3fv(glGetUniformLocation(computeProgram, "bgGridMin"), 1, glm::value_ptr(bgGridMin));
        glUniform3fv(glGetUniformLocation(computeProgram, "bgGridMax"), 1, glm::value_ptr(bgGridMax));
        glUniform3i(glGetUniformLocation(computeProgram, "bgRes"), x_res, y_res, z_res);

        // Dispatch compute shader
        glDispatchCompute(groupsX, groupsY, groupsZ);

        // Memory barrier to ensure compute shader finished writing
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // Surface detection pass (runs once after all objects are processed)
    glUseProgram(surfaceComputeProgram);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, backgroundDensitiesSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, surfaceDensitiesSSBO);
    glUniform3i(glGetUniformLocation(surfaceComputeProgram, "bgRes"), x_res, y_res, z_res);

    glDispatchCompute(groupsX, groupsY, groupsZ);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Update fluid obstacles after voxel collision update
    if (fluidInitialized) {
        updateFluidObstacles();
    }
}

// Single object overload for backwards compatibility
void get_background_points_GPU(voxel_object& v) {
    std::vector<voxel_object> temp = { v };
    // Temporarily set up GPU data if needed
    if (voxelObjectGPUData.empty()) {
        // This path shouldn't normally be hit with the new code
        cerr << "Warning: get_background_points_GPU called with single object but no GPU data" << endl;
        return;
    }
    get_background_points_GPU(voxel_objects);
}



// ============================================================================
// Read back surface points for rendering
// ============================================================================

void updateSurfacePointsForRendering(voxel_object& v) {
    if (!gpuInitialized) return;

    size_t gridSize = x_res * y_res * z_res;

    // Read back surface densities
    vector<float> surfaceDensities(gridSize);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, surfaceDensitiesSSBO);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * sizeof(float), surfaceDensities.data());

    // Build surface point vertices
    vector<RenderVertex> surfaceVertices;
    surfaceVertices.reserve(gridSize / 10); // Estimate

    float x_grid_min = 0;
    float y_grid_min = 0;
    float z_grid_min = 0;

    float x_step = (x_grid_max - x_grid_min) / (x_res - 1);
    float y_step = (y_grid_max - y_grid_min) / (y_res - 1);
    float z_step = (z_grid_max - z_grid_min) / (z_res - 1);

    for (size_t z = 0; z < z_res; z++) {
        for (size_t y = 0; y < y_res; y++) {
            for (size_t x = 0; x < x_res; x++) {
                size_t idx = x + y * x_res + z * x_res * y_res;
                if (surfaceDensities[idx] > 0.0f) {
                    RenderVertex rv;
                    rv.position[0] = x_grid_min + x * x_step;
                    rv.position[1] = y_grid_min + y * y_step;
                    rv.position[2] = z_grid_min + z * z_step;
                    rv.color[0] = 0.0f;
                    rv.color[1] = 1.0f;
                    rv.color[2] = 1.0f;
                    surfaceVertices.push_back(rv);
                }
            }
        }
    }

    numSurfacePoints = surfaceVertices.size();

    // Update point VBO
    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
    glBufferData(GL_ARRAY_BUFFER, surfaceVertices.size() * sizeof(RenderVertex),
        surfaceVertices.empty() ? nullptr : surfaceVertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

// ============================================================================
// Optimized Drawing Functions
// ============================================================================

void draw_triangles_fast(void) {
    if (numTriangleIndices == 0 || renderProgram == 0) return;

    glUseProgram(renderProgram);

    glm::mat4 identity(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(renderProgram, "model"), 1, GL_FALSE, glm::value_ptr(identity));
    glUniformMatrix4fv(glGetUniformLocation(renderProgram, "view"), 1, GL_FALSE, glm::value_ptr(main_camera.view_mat));
    glUniformMatrix4fv(glGetUniformLocation(renderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(main_camera.projection_mat));

    // Set camera position for specular
    glUniform3fv(glGetUniformLocation(renderProgram, "viewPos"), 1, &main_camera.eye.x);

    // Set ambient light
    glUniform3f(glGetUniformLocation(renderProgram, "ambientColor"), 1.0f, 1.0f, 1.0f);
    glUniform1f(glGetUniformLocation(renderProgram, "ambientStrength"), 0.5f);

    // Set point light data
// Set point light data - only for ACTUAL lights, not MAX
    int numLights = static_cast<int>(pointLights.size());  // Use actual size
    numLights = std::min(numLights, MAX_POINT_LIGHTS);     // Cap at shader limit
    glUniform1i(glGetUniformLocation(renderProgram, "numPointLights"), numLights);



    for (int i = 0; i < numLights; ++i) {
        std::string prefix = "lightPositions[" + std::to_string(i) + "]";
        glUniform3fv(glGetUniformLocation(renderProgram, prefix.c_str()), 1, glm::value_ptr(pointLights[i].position));

        prefix = "lightIntensities[" + std::to_string(i) + "]";
        glUniform1f(glGetUniformLocation(renderProgram, prefix.c_str()), pointLights[i].intensity);

        prefix = "lightColors[" + std::to_string(i) + "]";
        glUniform3fv(glGetUniformLocation(renderProgram, prefix.c_str()), 1, glm::value_ptr(pointLights[i].color));

        prefix = "lightFarPlanes[" + std::to_string(i) + "]";
        glUniform1f(glGetUniformLocation(renderProgram, prefix.c_str()), pointLights[i].farPlane);

        // NEW: Pass enabled state (as int: 1 or 0)
        prefix = "lightEnabled[" + std::to_string(i) + "]";
        glUniform1i(glGetUniformLocation(renderProgram, prefix.c_str()), pointLights[i].enabled ? 1 : 0);

        // Bind shadow cubemap
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_CUBE_MAP, pointLights[i].depthCubemap);
        prefix = "shadowMaps[" + std::to_string(i) + "]";
        glUniform1i(glGetUniformLocation(renderProgram, prefix.c_str()), i);
    }



    glBindVertexArray(triangleVAO);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(numTriangleIndices), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // Unbind shadow maps
    for (int i = 0; i < numLights; ++i) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    }
}




void draw_points_fast() {
    if (numSurfacePoints == 0 || renderProgram == 0) return;

    glUseProgram(renderProgram);

    glm::mat4 identity(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(renderProgram, "model"), 1, GL_FALSE, glm::value_ptr(identity));
    glUniformMatrix4fv(glGetUniformLocation(renderProgram, "view"), 1, GL_FALSE, glm::value_ptr(main_camera.view_mat));
    glUniformMatrix4fv(glGetUniformLocation(renderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(main_camera.projection_mat));

    glPointSize(3.0f);
    glBindVertexArray(pointVAO);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(numSurfacePoints));
    glBindVertexArray(0);
}

void draw_axis_fast() {
    if (renderProgram == 0) return;

    glUseProgram(renderProgram);

    glm::mat4 identity(1.0f);
    glUniformMatrix4fv(glGetUniformLocation(renderProgram, "model"), 1, GL_FALSE, glm::value_ptr(identity));
    glUniformMatrix4fv(glGetUniformLocation(renderProgram, "view"), 1, GL_FALSE, glm::value_ptr(main_camera.view_mat));
    glUniformMatrix4fv(glGetUniformLocation(renderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(main_camera.projection_mat));

    glBindVertexArray(axisVAO);
    glDrawArrays(GL_LINES, 0, 12);
    glBindVertexArray(0);
}

// ============================================================================
// Screenshot functionality
// ============================================================================

bool screenshot_mode = false;

void take_screenshot(size_t num_cams_wide, const char* filename, const bool reverse_rows = false)
{
    screenshot_mode = true;

    unsigned char  idlength = 0;
    unsigned char  colourmaptype = 0;
    unsigned char  datatypecode = 2;
    unsigned short int colourmaporigin = 0;
    unsigned short int colourmaplength = 0;
    unsigned char  colourmapdepth = 0;
    unsigned short int x_origin = 0;
    unsigned short int y_origin = 0;

    cout << "Image size: " << static_cast<size_t>(win_x) * num_cams_wide << "x" << static_cast<size_t>(win_y) * num_cams_wide << " pixels" << endl;

    if (static_cast<size_t>(win_x) * num_cams_wide > static_cast<unsigned short>(-1) ||
        static_cast<size_t>(win_y) * num_cams_wide > static_cast<unsigned short>(-1))
    {
        cout << "Image too large. Maximum width and height is " << static_cast<unsigned short>(-1) << endl;
        return;
    }

    unsigned short int px = win_x * static_cast<unsigned short>(num_cams_wide);
    unsigned short int py = win_y * static_cast<unsigned short>(num_cams_wide);
    unsigned char  bitsperpixel = 24;
    unsigned char  imagedescriptor = 0;
    vector<char> buffer(80, 0);

    size_t num_bytes = 3 * px * py;
    vector<unsigned char> pixel_data(num_bytes);
    vector<unsigned char> fbpixels(3 * win_x * win_y);

    const size_t total_cams = num_cams_wide * num_cams_wide;
    size_t cam_count = 0;

    for (size_t cam_num_x = 0; cam_num_x < num_cams_wide; cam_num_x++)
    {
        for (size_t cam_num_y = 0; cam_num_y < num_cams_wide; cam_num_y++)
        {
            cout << "Camera: " << cam_count + 1 << " of " << total_cams << endl;
            main_camera.Set_Large_Screenshot(num_cams_wide, cam_num_x, cam_num_y, win_x, win_y);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            // Draw objects for screenshot
            draw_points_fast();
            if (draw_triangles_on_screen) {
                draw_triangles_fast();
            }

            if (fluidSimEnabled) {
                draw_fluid_fast();
            }

            if (draw_axis) {
                draw_axis_fast();
            }
            glFlush();
            glReadPixels(0, 0, win_x, win_y, GL_RGB, GL_UNSIGNED_BYTE, &fbpixels[0]);

            for (GLint i = 0; i < win_x; i++)
            {
                for (GLint j = 0; j < win_y; j++)
                {
                    size_t fb_index = 3 * (j * win_x + i);
                    size_t screenshot_x = cam_num_x * win_x + i;
                    size_t screenshot_y = cam_num_y * win_y + j;
                    size_t screenshot_index = 3 * (screenshot_y * (win_x * num_cams_wide) + screenshot_x);

                    pixel_data[screenshot_index] = fbpixels[fb_index + 2];
                    pixel_data[screenshot_index + 1] = fbpixels[fb_index + 1];
                    pixel_data[screenshot_index + 2] = fbpixels[fb_index];
                }
            }
            cam_count++;
        }
    }

    screenshot_mode = false;
    main_camera.calculate_camera_matrices(win_x, win_y);

    ofstream out(filename, ios::binary);
    if (!out.is_open())
    {
        cout << "Failed to open TGA file for writing: " << filename << endl;
        return;
    }

    out.write(reinterpret_cast<char*>(&idlength), 1);
    out.write(reinterpret_cast<char*>(&colourmaptype), 1);
    out.write(reinterpret_cast<char*>(&datatypecode), 1);
    out.write(reinterpret_cast<char*>(&colourmaporigin), 2);
    out.write(reinterpret_cast<char*>(&colourmaplength), 2);
    out.write(reinterpret_cast<char*>(&colourmapdepth), 1);
    out.write(reinterpret_cast<char*>(&x_origin), 2);
    out.write(reinterpret_cast<char*>(&y_origin), 2);
    out.write(reinterpret_cast<char*>(&px), 2);
    out.write(reinterpret_cast<char*>(&py), 2);
    out.write(reinterpret_cast<char*>(&bitsperpixel), 1);
    out.write(reinterpret_cast<char*>(&imagedescriptor), 1);
    out.write(reinterpret_cast<char*>(&pixel_data[0]), num_bytes);
}

// ============================================================================
// GLUT Callbacks
// ============================================================================


void init_opengl(const int& width, const int& height)
{
    win_x = width;
    win_y = height;

    if (win_x < 1) win_x = 1;
    if (win_y < 1) win_y = 1;

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(win_x, win_y);
    win_id = glutCreateWindow("GPU Fluid Simulation with Temperature, Buoyancy & Gravity");

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    // Enable blending for fluid visualization
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glClearColor((float)background_colour.x, (float)background_colour.y, (float)background_colour.z, 1);
    glClearDepth(1.0f);

    main_camera.calculate_camera_matrices(win_x, win_y);
}

void reshape_func(int width, int height)
{
    win_x = width;
    win_y = height;

    if (win_x < 1) win_x = 1;
    if (win_y < 1) win_y = 1;

    glutSetWindow(win_id);
    glutReshapeWindow(win_x, win_y);
    glViewport(0, 0, win_x, win_y);

    main_camera.calculate_camera_matrices(win_x, win_y);

    if (textRenderer)
        textRenderer->setProjection(win_x, win_y);

    //// Save light positions/settings before cleanup
    //std::vector<glm::vec3> savedPositions;
    //std::vector<float> savedIntensities;
    //std::vector<glm::vec3> savedColors;
    //std::vector<bool> savedEnabled;

    //for (const auto& light : pointLights) {
    //    savedPositions.push_back(light.position);
    //    savedIntensities.push_back(light.intensity);
    //    savedColors.push_back(light.color);
    //    savedEnabled.push_back(light.enabled);
    //}

    cleanupShadowMaps();
    initShadowMaps();

    //// Restore additional lights if there were more than 1
    //for (size_t i = 0; i < savedPositions.size() && i < MAX_POINT_LIGHTS; i++) {
    //    addPointLight(savedPositions[i], savedIntensities[i], savedColors[i]);
    //    pointLights.back().enabled = savedEnabled[i];
    //}

}

void draw_objects(void)
{
    renderShadowMaps();

    // Draw surface points (voxel boundaries)
  //  draw_points_fast();

    // Draw triangles (voxel mesh)
    if (draw_triangles_on_screen) {
        draw_triangles_fast();
    }

    // Draw fluid
    if (fluidSimEnabled) {
        draw_fluid_fast();
    }

    // Draw axes
    if (draw_axis) {
        draw_axis_fast();
    }
}

float GLOBAL_TIME = 0;




void idle_func(void) {
    if (fluidSimEnabled && fluidInitialized) {
        static float accumulator = 0.0f;
        static float lastTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

        float currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
        float frameDelta = currentTime - lastTime;
        lastTime = currentTime;

        accumulator += frameDelta;

        // Optional: Cap accumulator to prevent spiral of death if FPS drops too low
        if (accumulator > 0.2f) accumulator = 0.2f;  // Max ~6 steps per frame

        while (accumulator >= fluidParams.dt) {
            stepFluidSimulation();



            GLOBAL_TIME += fluidParams.dt;
            accumulator -= fluidParams.dt;
        }


        size_t enabled_count = 0;

        for (size_t i = 0; i < pointLights.size(); i++)
            if (pointLights[i].enabled)
                enabled_count++;

        if (injectDensity || injectVelocity) {
            mouseVelocity = (currentMouseWorldPos - lastMouseWorldPos) * 10.0f;
            addFluidSource(currentMouseWorldPos, mouseVelocity, injectDensity, injectVelocity);
            lastMouseWorldPos = currentMouseWorldPos;

            currentMouseWorldPos.z = z_grid_max / 2.0;

            // Use a fixed secondary light slot instead of creating new lights
            if (pointLights.size() >= 2) {
                // Reuse existing light at index 1
                pointLights[1].position = currentMouseWorldPos;
                pointLights[1].enabled = true;
            }
            // Note: Don't create new lights during runtime - manage them at init
        }
        else
        {
            // Disable the injection light when not injecting
            if (pointLights.size() >= 2) {
                pointLights[1].enabled = false;
            }
        }


        // Detect fluid-obstacle collisions
        static const int COLLISION_INTERVAL_MS = 100;
        static int collision_lastCallTime = 0;
        int curr_time_int = glutGet(GLUT_ELAPSED_TIME);

        if (curr_time_int - collision_lastCallTime >= COLLISION_INTERVAL_MS)
        {
            for (auto& vo : voxel_objects)
                do_blackening(vo);

            collision_lastCallTime = curr_time_int;
        }
    }

    glutPostRedisplay();
}



void display_func(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    draw_objects();
    displayFPS();
    glFlush();

    if (false == screenshot_mode)
        glutSwapBuffers();
}

void keyboard_func(unsigned char key, int x, int y)
{
    switch (tolower(key))
    {
    case 'i':  // Toggle between marching cubes and ray marching
        useMarchingCubes = !useMarchingCubes;
        cout << "Rendering mode: " << (useMarchingCubes ? "Marching Cubes" : "Ray Marching") << endl;
        break;

    case 'm':
        take_screenshot(4, "out.tga");
        break;

    case 't':
        draw_triangles_on_screen = !draw_triangles_on_screen;
        break;

    case 'w':
        draw_axis = !draw_axis;
        break;

    case 'e':
        draw_control_list = !draw_control_list;
        break;

        // Toggle fluid simulation
    case 'f':
        fluidSimEnabled = !fluidSimEnabled;
        cout << "Fluid simulation: " << (fluidSimEnabled ? "ON" : "OFF") << endl;
        break;

        // Reset fluid
    //case 'c':
    //    if (fluidInitialized) {
    //        size_t gridSize = x_res * y_res * z_res;
    //        vector<float> zeroDensity(gridSize, 0.0f);
    //        vector<float> ambientTemp(gridSize, fluidParams.ambientTemperature);
    //        vector<glm::vec4> zeroVelocity(gridSize, glm::vec4(0.0f));

    //        for (int i = 0; i < 2; i++) {
    //            glBindBuffer(GL_SHADER_STORAGE_BUFFER, densitySSBO[i]);
    //            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * sizeof(float), zeroDensity.data());

    //            glBindBuffer(GL_SHADER_STORAGE_BUFFER, velocitySSBO[i]);
    //            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * sizeof(glm::vec4), zeroVelocity.data());

    //            // Reset temperature to ambient
    //            glBindBuffer(GL_SHADER_STORAGE_BUFFER, temperatureSSBO[i]);
    //            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * sizeof(float), ambientTemp.data());
    //        }

    //        // Reset blackening state
    //        if (!vo.voxel_original_colours.empty()) {
    //            vo.voxel_colours = vo.voxel_original_colours;
    //            fill(vo.voxel_blacken_times.begin(), vo.voxel_blacken_times.end(), -1.0f);
    //            vo.tri_vec.clear();
    //            get_triangles(vo.tri_vec, vo);
    //            updateTriangleBuffer(vo);
    //        }

    //        simulationStartTime = std::chrono::steady_clock::now();  // Reset clock

    //        cout << "Fluid reset (including temperature and blackening)" << endl;
    //    }
    //    break;

        // Adjust viscosity
    case '[':
        fluidParams.viscosity *= 0.5f;
        cout << "Viscosity: " << fluidParams.viscosity << endl;
        break;
    case ']':
        fluidParams.viscosity *= 2.0f;
        cout << "Viscosity: " << fluidParams.viscosity << endl;
        break;

        // Adjust injection radius
    case ',':
        injectRadius = std::max(1, injectRadius - 1);
        cout << "Injection radius: " << injectRadius << endl;
        break;
    case '.':
        injectRadius = std::min(10, injectRadius + 1);
        cout << "Injection radius: " << injectRadius << endl;
        break;

        // ========================================
        // NEW: Temperature and Buoyancy Controls
        // ========================================

        // Toggle gravity
    case 'g':
        fluidParams.enableGravity = !fluidParams.enableGravity;
        cout << "Gravity: " << (fluidParams.enableGravity ? "ON" : "OFF") << endl;
        break;

        // Toggle buoyancy
    case 'y':
        fluidParams.enableBuoyancy = !fluidParams.enableBuoyancy;
        cout << "Buoyancy: " << (fluidParams.enableBuoyancy ? "ON" : "OFF") << endl;
        break;

        // Toggle temperature visualization
    case 'v':
        fluidParams.visualizeTemperature = !fluidParams.visualizeTemperature;
        cout << "Visualize temperature: " << (fluidParams.visualizeTemperature ? "ON" : "OFF") << endl;
        break;

        // Adjust buoyancy alpha (density sinking)
    case '1':
        fluidParams.buoyancyAlpha = std::max(0.0f, fluidParams.buoyancyAlpha - 0.01f);
        cout << "Buoyancy Alpha (density): " << fluidParams.buoyancyAlpha << endl;
        break;
    case '2':
        fluidParams.buoyancyAlpha = std::min(1.0f, fluidParams.buoyancyAlpha + 0.01f);
        cout << "Buoyancy Alpha (density): " << fluidParams.buoyancyAlpha << endl;
        break;

        // Adjust buoyancy beta (temperature rising)
    case '3':
        fluidParams.buoyancyBeta = std::max(0.0f, fluidParams.buoyancyBeta - 0.1f);
        cout << "Buoyancy Beta (temperature): " << fluidParams.buoyancyBeta << endl;
        break;
    case '4':
        fluidParams.buoyancyBeta = std::min(5.0f, fluidParams.buoyancyBeta + 0.1f);
        cout << "Buoyancy Beta (temperature): " << fluidParams.buoyancyBeta << endl;
        break;

        // Adjust temperature amount
    case '5':
        fluidParams.temperatureAmount = std::max(0.0f, fluidParams.temperatureAmount - 1.0f);
        cout << "Temperature injection amount: " << fluidParams.temperatureAmount << endl;
        break;
    case '6':
        fluidParams.temperatureAmount = std::min(50.0f, fluidParams.temperatureAmount + 1.0f);
        cout << "Temperature injection amount: " << fluidParams.temperatureAmount << endl;
        break;

        // Adjust gravity strength
    case '7':
        fluidParams.gravity = std::max(0.0f, fluidParams.gravity - 1.0f);
        cout << "Gravity: " << fluidParams.gravity << endl;
        break;
    case '8':
        fluidParams.gravity = std::min(50.0f, fluidParams.gravity + 1.0f);
        cout << "Gravity: " << fluidParams.gravity << endl;
        break;

        // ====================================================================
        // VOLUME LIGHTING CONTROLS
        // ====================================================================
    case '9':
        fluidParams.phaseG = std::max(-0.99f, fluidParams.phaseG - 0.1f);
        cout << "Phase G (scattering): " << fluidParams.phaseG << " (negative = back scatter)" << endl;
        break;
    case '0':
        fluidParams.phaseG = std::min(0.99f, fluidParams.phaseG + 0.1f);
        cout << "Phase G (scattering): " << fluidParams.phaseG << " (positive = forward scatter)" << endl;
        break;
    case 'l':
    case 'L':
        fluidParams.enableVolumeLighting = !fluidParams.enableVolumeLighting;
        cout << "Volume lighting: " << (fluidParams.enableVolumeLighting ? "ON" : "OFF") << endl;
        break;
    case 'k':
    case 'K':
        fluidParams.enableVolumeShadows = !fluidParams.enableVolumeShadows;
        cout << "Volume shadows: " << (fluidParams.enableVolumeShadows ? "ON" : "OFF") << endl;
        break;
    case 'j':
    case 'J':
        fluidParams.volumeScattering = std::max(0.0f, fluidParams.volumeScattering - 0.1f);
        cout << "Volume scattering: " << fluidParams.volumeScattering << endl;
        break;
    case 'u':
    case 'U':
        fluidParams.volumeScattering = std::min(2.0f, fluidParams.volumeScattering + 0.1f);
        cout << "Volume scattering: " << fluidParams.volumeScattering << endl;
        break;

    case 'o':
    {
        // Assuming you want to operate on a specifi c voxel object, e.g., voxel_objects[0]
        voxel_object& vo = voxel_objects[0];  // or whichever object you want to transform

        vo.u += 0.1f;
        vo.model_matrix = glm::mat4(1.0f);

        vo.model_matrix = glm::translate(vo.model_matrix, voxelFiles[0].location);

        vo.model_matrix = glm::rotate(vo.model_matrix, vo.u, glm::vec3(0.0f, 1.0f, 0.0f));
        vo.model_matrix = glm::rotate(vo.model_matrix, vo.v, glm::vec3(1.0f, 0.0f, 0.0f));

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        get_background_points_GPU(voxel_objects);
        updateTriangleBuffer(voxel_objects);

        //        updateSurfacePointsForRendering(vo);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> elapsed = end - start;
        cout << "GPU compute time: " << elapsed.count() << " ms" << endl;


        break;
    }













    //case 'o':
    //{
    //    vo.u += 0.1f;
    //    vo.model_matrix = glm::mat4(1.0f);
    //    vo.model_matrix = glm::translate(vo.model_matrix, knight_location);

    //    vo.model_matrix = glm::rotate(vo.model_matrix, vo.u, glm::vec3(0.0f, 1.0f, 0.0f));
    //    vo.model_matrix = glm::rotate(vo.model_matrix, vo.v, glm::vec3(1.0f, 0.0f, 0.0f));

    //    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    //    get_background_points_GPU(vo);
    //    glFinish();
    //    updateSurfacePointsForRendering(vo);
    //    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    //    std::chrono::duration<float, std::milli> elapsed = end - start;
    //    cout << "GPU compute time: " << elapsed.count() << " ms" << endl;
    //    break;
    //}
    //case 'p':
    //{
    //    vo.u -= 0.1f;
    //    vo.model_matrix = glm::mat4(1.0f);
    //    vo.model_matrix = glm::translate(vo.model_matrix, knight_location);

    //    vo.model_matrix = glm::rotate(vo.model_matrix, vo.u, glm::vec3(0.0f, 1.0f, 0.0f));
    //    vo.model_matrix = glm::rotate(vo.model_matrix, vo.v, glm::vec3(1.0f, 0.0f, 0.0f));

    //    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    //    get_background_points_GPU(vo);
    //    glFinish();
    //    updateSurfacePointsForRendering(vo);
    //    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    //    std::chrono::duration<float, std::milli> elapsed = end - start;
    //    cout << "GPU compute time: " << elapsed.count() << " ms" << endl;
    //    break;
    //}
    //case 'k':
    //{
    //    vo.v += 0.1f;
    //    vo.model_matrix = glm::mat4(1.0f);
    //    vo.model_matrix = glm::translate(vo.model_matrix, knight_location);

    //    vo.model_matrix = glm::rotate(vo.model_matrix, vo.u, glm::vec3(0.0f, 1.0f, 0.0f));
    //    vo.model_matrix = glm::rotate(vo.model_matrix, vo.v, glm::vec3(1.0f, 0.0f, 0.0f));

    //    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    //    get_background_points_GPU(vo);
    //    glFinish();
    //    updateSurfacePointsForRendering(vo);
    //    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    //    std::chrono::duration<float, std::milli> elapsed = end - start;
    //    cout << "GPU compute time: " << elapsed.count() << " ms" << endl;
    //    break;
    //}
    //case 'l':
    //{
    //    vo.v -= 0.1f;
    //    vo.model_matrix = glm::mat4(1.0f);
    //    vo.model_matrix = glm::translate(vo.model_matrix, knight_location);

    //    vo.model_matrix = glm::rotate(vo.model_matrix, vo.u, glm::vec3(0.0f, 1.0f, 0.0f));
    //    vo.model_matrix = glm::rotate(vo.model_matrix, vo.v, glm::vec3(1.0f, 0.0f, 0.0f));

    //    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    //    get_background_points_GPU(vo);
    //    glFinish();
    //    updateSurfacePointsForRendering(vo);
    //    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    //    std::chrono::duration<float, std::milli> elapsed = end - start;
    //    cout << "GPU compute time: " << elapsed.count() << " ms" << endl;
    //    break;
    //}



    // Print controls
    case 'h':
        cout << "\n=== CONTROLS ===" << endl;
        cout << "F: Toggle fluid simulation" << endl;
        cout << "C: Clear/reset fluid" << endl;
        cout << "Middle Mouse + Drag: Inject density and velocity" << endl;
        cout << "Shift + Middle Mouse: Inject density only" << endl;
        cout << "[/]: Decrease/increase viscosity" << endl;
        cout << "-/=: Decrease/increase turbulence (Smagorinsky constant)" << endl;
        cout << ",/.: Decrease/increase injection radius" << endl;
        cout << "\n=== NEW: TEMPERATURE & BUOYANCY ===" << endl;
        cout << "G: Toggle gravity" << endl;
        cout << "Y: Toggle buoyancy" << endl;
        cout << "V: Toggle temperature visualization" << endl;
        cout << "1/2: Decrease/increase buoyancy alpha (density sinking)" << endl;
        cout << "3/4: Decrease/increase buoyancy beta (temperature rising)" << endl;
        cout << "5/6: Decrease/increase temperature injection" << endl;
        cout << "7/8: Decrease/increase gravity strength" << endl;
        cout << "\n=== OTHER ===" << endl;
        cout << "O/P: Rotate voxel object Y-axis" << endl;
        cout << "K/L: Rotate voxel object X-axis" << endl;
        cout << "T: Toggle triangle mesh" << endl;
        cout << "W: Toggle axes" << endl;
        cout << "R: Toggle real-time rotation" << endl;
        cout << "M: Take screenshot" << endl;
        cout << "H: Show this help" << endl;
        cout << "================\n" << endl;
        break;

    default:
        break;
    }
}

void mouse_func(int button, int state, int x, int y)
{
    if (GLUT_LEFT_BUTTON == button) {
        lmb_down = (GLUT_DOWN == state);
    }
    else if (GLUT_MIDDLE_BUTTON == button) {
        mmb_down = (GLUT_DOWN == state);

        // Start/stop fluid injection
        if (mmb_down) {
            int modifiers = glutGetModifiers();
            if (modifiers & GLUT_ACTIVE_SHIFT) {
                // Shift + Middle = density only
                injectDensity = true;
                injectVelocity = false;
            }
            else {
                // Middle = both density and velocity
                injectDensity = true;
                injectVelocity = true;
            }
            currentMouseWorldPos = screenToWorld_ortho(x, y, 0.0f);
            lastMouseWorldPos = currentMouseWorldPos;
        }
        else {
            injectDensity = false;
            injectVelocity = false;
        }
    }
    else if (GLUT_RIGHT_BUTTON == button) {
        rmb_down = (GLUT_DOWN == state);
    }
}

void motion_func(int x, int y)
{
    int prev_mouse_x = mouse_x;
    int prev_mouse_y = mouse_y;

    mouse_x = x;
    mouse_y = y;

    int mouse_delta_x = mouse_x - prev_mouse_x;
    int mouse_delta_y = prev_mouse_y - mouse_y;

    if (lmb_down && (0 != mouse_delta_x || 0 != mouse_delta_y)) {
        main_camera.u -= static_cast<float>(mouse_delta_y) * u_spacer;
        main_camera.v += static_cast<float>(mouse_delta_x) * v_spacer;
    }
    else if (rmb_down && (0 != mouse_delta_y)) {
        main_camera.w -= static_cast<float>(mouse_delta_y) * w_spacer;
        if (main_camera.w < 1.1f)
            main_camera.w = 1.1f;
    }
    else if (mmb_down) {
        // Update mouse world position for fluid injection
        currentMouseWorldPos = screenToWorld_ortho(x, y, 0.0f);
    }

    main_camera.calculate_camera_matrices(win_x, win_y);
}

void passive_motion_func(int x, int y)
{
    mouse_x = x;
    mouse_y = y;
}

void cleanup(void)
{
    // Cleanup GPU resources
    if (computeProgram) glDeleteProgram(computeProgram);
    if (surfaceComputeProgram) glDeleteProgram(surfaceComputeProgram);
    if (renderProgram) glDeleteProgram(renderProgram);

    glDeleteBuffers(1, &voxelCentresSSBO);
    glDeleteBuffers(1, &voxelDensitiesSSBO);
    glDeleteBuffers(1, &gridMinMaxSSBO);
    glDeleteBuffers(1, &voGridCellsSSBO);
    glDeleteBuffers(1, &backgroundDensitiesSSBO);
    glDeleteBuffers(1, &backgroundCollisionsSSBO);
    glDeleteBuffers(1, &surfaceDensitiesSSBO);

    glDeleteVertexArrays(1, &triangleVAO);
    glDeleteBuffers(1, &triangleVBO);
    glDeleteBuffers(1, &triangleEBO);

    glDeleteVertexArrays(1, &pointVAO);
    glDeleteBuffers(1, &pointVBO);

    glDeleteVertexArrays(1, &axisVAO);
    glDeleteBuffers(1, &axisVBO);

    // Cleanup fluid resources
    if (fluidInitialized) {
        glDeleteBuffers(2, velocitySSBO);
        glDeleteBuffers(2, densitySSBO);
        glDeleteBuffers(2, pressureSSBO);
        glDeleteBuffers(1, &divergenceSSBO);
        glDeleteBuffers(1, &obstacleSSBO);
        glDeleteBuffers(1, &turbulentViscositySSBO);

        // NEW: Cleanup temperature buffers
        glDeleteBuffers(2, temperatureSSBO);

        glDeleteVertexArrays(1, &fluidVAO);
        glDeleteBuffers(1, &fluidVBO);

        if (advectionProgram) glDeleteProgram(advectionProgram);
        if (diffusionProgram) glDeleteProgram(diffusionProgram);
        if (divergenceProgram) glDeleteProgram(divergenceProgram);
        if (pressureProgram) glDeleteProgram(pressureProgram);
        if (gradientSubtractProgram) glDeleteProgram(gradientSubtractProgram);
        if (boundaryProgram) glDeleteProgram(boundaryProgram);
        if (addSourceProgram) glDeleteProgram(addSourceProgram);
        if (turbulenceProgram) glDeleteProgram(turbulenceProgram);
        if (obstacleProgram) glDeleteProgram(obstacleProgram);
        if (buoyancyProgram) glDeleteProgram(buoyancyProgram);
    }

    cleanupMarchingCubes();

    cleanupShadowMaps();


    glutDestroyWindow(win_id);
}

// ============================================================================
// Main entry point
// ============================================================================

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    init_opengl(win_x, win_y);

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        cerr << "Error: " << glewGetErrorString(err) << endl;
        return 1;
    }

    // Check for compute shader support
    if (!GLEW_ARB_compute_shader) {
        cerr << "Error: Compute shaders not supported!" << endl;
        return 1;
    }

    cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
    cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;


    // Load all voxel objects
    voxel_objects.resize(voxelFiles.size());

    for (size_t i = 0; i < voxelFiles.size(); i++) {
        cout << "Loading voxel file: " << voxelFiles[i].filename << endl;

        // Set up model matrix with location
        voxel_objects[i].model_matrix = glm::mat4(1.0f);
        voxel_objects[i].model_matrix = glm::translate(voxel_objects[i].model_matrix, voxelFiles[i].location);

        // Load voxel data
        if (!get_voxels(voxelFiles[i].filename.c_str(), voxel_objects[i])) {
            cerr << "Failed to load: " << voxelFiles[i].filename << endl;
            continue;
        }

        // Generate triangles
        get_triangles(voxel_objects[i].tri_vec, voxel_objects[i]);

        cout << "Loaded " << voxelFiles[i].filename << " at position ("
            << voxelFiles[i].location.x << ", "
            << voxelFiles[i].location.y << ", "
            << voxelFiles[i].location.z << ")" << endl;
    }

    // Initialize GPU buffers for all voxel objects
    initGPUBuffers(voxel_objects);

    // Initial GPU computation for all objects
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    get_background_points_GPU(voxel_objects);
    glFinish(); // Wait for GPU to complete
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end - start;
    cout << "Initial GPU background points computation: " << elapsed.count() << " ms" << endl;

    // Update render buffers with all objects
    updateTriangleBuffer(voxel_objects);
    //updateSurfacePointsForRendering(voxel_objects[0]); // Surface points are shared


    cout << "Surface points: " << numSurfacePoints << endl;

    // Initialize fluid simulation
    initFluidSimulation();

    initShadowMaps();

    initDefaultLights();

    // Update obstacles from voxel collisions
    updateFluidObstacles();

    // Print controls
    cout << "\n=== FLUID SIMULATION WITH TEMPERATURE, BUOYANCY & GRAVITY ===" << endl;
    cout << "F: Toggle fluid simulation" << endl;
    cout << "C: Clear/reset fluid" << endl;
    cout << "Middle Mouse + Drag: Inject density, velocity, and temperature" << endl;
    cout << "Shift + Middle Mouse: Inject density and temperature only" << endl;
    cout << "[/]: Decrease/increase viscosity" << endl;
    cout << "-/=: Decrease/increase turbulence" << endl;
    cout << ",/.: Decrease/increase injection radius" << endl;
    cout << "\n--- Temperature & Buoyancy Controls ---" << endl;
    cout << "G: Toggle gravity (currently " << (fluidParams.enableGravity ? "ON" : "OFF") << ")" << endl;
    cout << "Y: Toggle buoyancy (currently " << (fluidParams.enableBuoyancy ? "ON" : "OFF") << ")" << endl;
    cout << "V: Toggle temperature visualization" << endl;
    cout << "1/2: Adjust buoyancy alpha (density)" << endl;
    cout << "3/4: Adjust buoyancy beta (temperature)" << endl;
    cout << "5/6: Adjust temperature injection amount" << endl;
    cout << "7/8: Adjust gravity strength" << endl;
    cout << "\n--- Volume Lighting Controls ---" << endl;
    cout << "9/0: Adjust phase function (scattering direction)" << endl;
    cout << "L: Toggle volume lighting" << endl;
    cout << "K: Toggle volume shadows" << endl;
    cout << "\nH: Show all controls" << endl;
    cout << "=================================\n" << endl;

    glutReshapeFunc(reshape_func);
    glutIdleFunc(idle_func);
    glutDisplayFunc(display_func);
    glutKeyboardFunc(keyboard_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutPassiveMotionFunc(passive_motion_func);



    textRenderer = new TextRenderer("font.png", win_x, win_y);


    // Start fluid simulation timer
    //glutTimerFunc(1, fluid_timer_func, 0);

    glutMainLoop();

    return 0;
}