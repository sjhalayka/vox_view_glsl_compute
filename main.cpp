#include "main.h"
#include "shader_utils.h"

// ============================================================================
// GPU Compute Shader Implementation for Real-Time Voxel Grid Collision
// Uses OpenGL 4.3 Compute Shaders, GLEW, GLUT, GLM
// ============================================================================

// Vertex structure for shader compatibility
struct RenderVertex {
    float position[3];
    float color[3];
};

// GPU Buffer handles
GLuint computeProgram = 0;
GLuint surfaceComputeProgram = 0;

// SSBOs for compute shader
GLuint voxelCentresSSBO = 0;
GLuint voxelDensitiesSSBO = 0;
GLuint gridMinMaxSSBO = 0;
GLuint backgroundDensitiesSSBO = 0;
GLuint backgroundCollisionsSSBO = 0;
GLuint surfaceDensitiesSSBO = 0;
GLuint voGridCellsSSBO = 0;

// Persistent render buffers
GLuint triangleVAO = 0, triangleVBO = 0, triangleEBO = 0;
GLuint pointVAO = 0, pointVBO = 0;
GLuint axisVAO = 0, axisVBO = 0;
GLuint renderProgram = 0;

bool gpuInitialized = false;
size_t numSurfacePoints = 0;

// ============================================================================
// Compute Shader Sources
// ============================================================================

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
    
    if (voxelIndex >= 0) {
        backgroundDensities[index] = 1.0;
        backgroundCollisions[index] = voxelIndex;
    } else {
        backgroundDensities[index] = 0.0;
        backgroundCollisions[index] = -1;
    }
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

// Common vertex shader for all primitives
const char* commonVertexShaderSource = R"(
#version 430 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 fragColor;
void main() {
    fragColor = color;
    gl_Position = projection * view * model * vec4(position, 1.0);
}
)";

// Common fragment shader for all primitives
const char* commonFragmentShaderSource = R"(
#version 430 core
in vec3 fragColor;
out vec4 finalColor;
void main() {
    finalColor = vec4(fragColor, 1.0);
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
// GPU Initialization
// ============================================================================

void initGPUBuffers(voxel_object& v) {
    // Compile compute shaders
    computeProgram = compileComputeShader(backgroundPointsComputeShader);
    surfaceComputeProgram = compileComputeShader(surfaceDetectionComputeShader);

    if (computeProgram == 0 || surfaceComputeProgram == 0) {
        cerr << "Failed to compile compute shaders!" << endl;
        return;
    }

    // Create render program once
    renderProgram = createShaderProgram(commonVertexShaderSource, commonFragmentShaderSource);

    // Prepare voxel centres as vec4 array
    size_t numVoxels = v.voxel_centres.size();
    vector<glm::vec4> voxelCentresVec4(numVoxels);
    for (size_t i = 0; i < numVoxels; i++) {
        voxelCentresVec4[i] = glm::vec4(v.voxel_centres[i].x, v.voxel_centres[i].y, v.voxel_centres[i].z, 0.0f);
    }

    // Create SSBOs for voxel data
    glGenBuffers(1, &voxelCentresSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, voxelCentresSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numVoxels * sizeof(glm::vec4), voxelCentresVec4.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &voxelDensitiesSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, voxelDensitiesSSBO);
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

    glGenBuffers(1, &gridMinMaxSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridMinMaxSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GridMinMax), &gridData, GL_STATIC_DRAW);

    // Create vo_grid_cells SSBO (maps cell position to voxel index)
    vector<int> gridCellsInt(v.vo_grid_cells.size());
    for (size_t i = 0; i < v.vo_grid_cells.size(); i++) {
        gridCellsInt[i] = static_cast<int>(v.vo_grid_cells[i]);
    }

    glGenBuffers(1, &voGridCellsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, voGridCellsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, gridCellsInt.size() * sizeof(int), gridCellsInt.data(), GL_STATIC_DRAW);

    // Create output SSBOs for background grid
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
    cout << "GPU buffers initialized successfully" << endl;
}

// ============================================================================
// GPU Background Points Computation
// ============================================================================

void get_background_points_GPU(voxel_object& v) {
    if (!gpuInitialized) return;

    glUseProgram(computeProgram);

    // Bind SSBOs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, voxelCentresSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, voxelDensitiesSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gridMinMaxSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, backgroundDensitiesSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, backgroundCollisionsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, voGridCellsSSBO);

    // Set uniforms
    glm::mat4 invModel = glm::inverse(v.model_matrix);
    glUniformMatrix4fv(glGetUniformLocation(computeProgram, "invModelMatrix"), 1, GL_FALSE, glm::value_ptr(invModel));

    glm::vec3 bgGridMin(-x_grid_max, -y_grid_max, -z_grid_max);
    glm::vec3 bgGridMax(x_grid_max, y_grid_max, z_grid_max);
    glUniform3fv(glGetUniformLocation(computeProgram, "bgGridMin"), 1, glm::value_ptr(bgGridMin));
    glUniform3fv(glGetUniformLocation(computeProgram, "bgGridMax"), 1, glm::value_ptr(bgGridMax));
    glUniform3i(glGetUniformLocation(computeProgram, "bgRes"), x_res, y_res, z_res);

    // Dispatch compute shader
    GLuint groupsX = (x_res + 7) / 8;
    GLuint groupsY = (y_res + 7) / 8;
    GLuint groupsZ = (z_res + 7) / 8;
    glDispatchCompute(groupsX, groupsY, groupsZ);

    // Memory barrier to ensure compute shader finished writing
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Surface detection pass
    glUseProgram(surfaceComputeProgram);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, backgroundDensitiesSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, surfaceDensitiesSSBO);
    glUniform3i(glGetUniformLocation(surfaceComputeProgram, "bgRes"), x_res, y_res, z_res);

    glDispatchCompute(groupsX, groupsY, groupsZ);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
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

    float x_grid_min = -x_grid_max;
    float y_grid_min = -y_grid_max;
    float z_grid_min = -z_grid_max;

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
// Update triangle buffer for rendering
// ============================================================================

size_t numTriangleIndices = 0;

void updateTriangleBuffer(voxel_object& v) {
    if (!gpuInitialized) return;

    vector<RenderVertex> vertices;
    vector<GLuint> indices;

    for (size_t i = 0; i < v.tri_vec.size(); i++) {
        for (size_t j = 0; j < 3; j++) {
            RenderVertex rv;
            rv.position[0] = v.tri_vec[i].vertex[j].x;
            rv.position[1] = v.tri_vec[i].vertex[j].y;
            rv.position[2] = v.tri_vec[i].vertex[j].z;
            rv.color[0] = v.tri_vec[i].colour.x;
            rv.color[1] = v.tri_vec[i].colour.y;
            rv.color[2] = v.tri_vec[i].colour.z;
            vertices.push_back(rv);
            indices.push_back(static_cast<GLuint>(vertices.size() - 1));
        }
    }

    numTriangleIndices = indices.size();

    glBindVertexArray(triangleVAO);

    glBindBuffer(GL_ARRAY_BUFFER, triangleVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(RenderVertex),
        vertices.empty() ? nullptr : vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangleEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint),
        indices.empty() ? nullptr : indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

// ============================================================================
// Optimized Drawing Functions
// ============================================================================

void draw_triangles_fast(const glm::mat4& model) {
    if (numTriangleIndices == 0 || renderProgram == 0) return;

    glUseProgram(renderProgram);

    glUniformMatrix4fv(glGetUniformLocation(renderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(renderProgram, "view"), 1, GL_FALSE, glm::value_ptr(main_camera.view_mat));
    glUniformMatrix4fv(glGetUniformLocation(renderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(main_camera.projection_mat));

    glBindVertexArray(triangleVAO);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(numTriangleIndices), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
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
// Legacy drawing functions (kept for compatibility)
// ============================================================================

void draw_triangles(const std::vector<custom_math::vertex_3>& positions, const std::vector<custom_math::vertex_3>& colors, glm::mat4 model) {
    if (positions.empty() || colors.empty() || positions.size() != colors.size()) {
        return;
    }

    std::vector<GLuint> indices;
    for (size_t i = 0; i < positions.size(); i += 3) {
        if (i + 2 < positions.size()) {
            indices.push_back(static_cast<GLuint>(i));
            indices.push_back(static_cast<GLuint>(i + 1));
            indices.push_back(static_cast<GLuint>(i + 2));
        }
    }

    std::vector<RenderVertex> vertices;
    for (size_t i = 0; i < positions.size(); ++i) {
        RenderVertex vertex;
        vertex.position[0] = positions[i].x;
        vertex.position[1] = positions[i].y;
        vertex.position[2] = positions[i].z;
        vertex.color[0] = colors[i].x;
        vertex.color[1] = colors[i].y;
        vertex.color[2] = colors[i].z;
        vertices.push_back(vertex);
    }

    GLuint shaderProgram = createShaderProgram(commonVertexShaderSource, commonFragmentShaderSource);
    if (shaderProgram == 0) return;

    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(RenderVertex), vertices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glUseProgram(shaderProgram);

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(main_camera.view_mat));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(main_camera.projection_mat));

    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shaderProgram);
}

void draw_lines(const std::vector<custom_math::vertex_3>& positions, const std::vector<custom_math::vertex_3>& colors, glm::mat4 model) {
    if (positions.empty() || colors.empty() || positions.size() != colors.size()) {
        return;
    }

    std::vector<RenderVertex> vertices;
    for (size_t i = 0; i < positions.size(); ++i) {
        RenderVertex vertex;
        vertex.position[0] = positions[i].x;
        vertex.position[1] = positions[i].y;
        vertex.position[2] = positions[i].z;
        vertex.color[0] = colors[i].x;
        vertex.color[1] = colors[i].y;
        vertex.color[2] = colors[i].z;
        vertices.push_back(vertex);
    }

    GLuint shaderProgram = createShaderProgram(commonVertexShaderSource, commonFragmentShaderSource);
    if (shaderProgram == 0) return;

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(RenderVertex), vertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glUseProgram(shaderProgram);

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(main_camera.view_mat));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(main_camera.projection_mat));

    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertices.size()));

    glBindVertexArray(0);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shaderProgram);
}

void draw_points(const std::vector<custom_math::vertex_3>& positions, const std::vector<custom_math::vertex_3>& colors, glm::mat4 model) {
    if (positions.empty() || colors.empty() || positions.size() != colors.size()) {
        return;
    }

    std::vector<RenderVertex> vertices;
    for (size_t i = 0; i < positions.size(); ++i) {
        RenderVertex vertex;
        vertex.position[0] = positions[i].x;
        vertex.position[1] = positions[i].y;
        vertex.position[2] = positions[i].z;
        vertex.color[0] = colors[i].x;
        vertex.color[1] = colors[i].y;
        vertex.color[2] = colors[i].z;
        vertices.push_back(vertex);
    }

    GLuint shaderProgram = createShaderProgram(commonVertexShaderSource, commonFragmentShaderSource);
    if (shaderProgram == 0) return;

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(RenderVertex), vertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glUseProgram(shaderProgram);

    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(main_camera.view_mat));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(main_camera.projection_mat));

    glPointSize(5.0f);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vertices.size()));

    glBindVertexArray(0);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shaderProgram);
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
            display_func();
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

    test_texture.resize(x_res * y_res * z_res, 0);
    for (size_t x = 0; x < x_res; x++) {
        for (size_t y = 0; y < y_res; y++) {
            for (size_t z = 0; z < z_res; z++) {
                const size_t voxel_index = x + y * x_res + z * x_res * y_res;
                if (y >= y_res / 2)
                    test_texture[voxel_index] = 255;
            }
        }
    }

    vo.model_matrix = glm::mat4(1.0f);
    get_voxels("chr_knight.vox", vo);
    get_triangles(vo.tri_vec, vo);

    // Initialize GPU buffers after loading voxel data
    initGPUBuffers(vo);

    // Initial GPU computation
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    get_background_points_GPU(vo);
    glFinish(); // Wait for GPU to complete
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end - start;
    cout << "Initial GPU background points computation: " << elapsed.count() << " ms" << endl;

    // Update render buffers
    updateTriangleBuffer(vo);
    updateSurfacePointsForRendering(vo);

    cout << "Surface points: " << numSurfacePoints << endl;

    glutReshapeFunc(reshape_func);
    glutIdleFunc(idle_func);
    glutDisplayFunc(display_func);
    glutKeyboardFunc(keyboard_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutPassiveMotionFunc(passive_motion_func);

    glutMainLoop();

    return 0;
}

// ============================================================================
// GLUT Callbacks
// ============================================================================

void idle_func(void)
{
    glutPostRedisplay();
}

void init_opengl(const int& width, const int& height)
{
    win_x = width;
    win_y = height;

    if (win_x < 1) win_x = 1;
    if (win_y < 1) win_y = 1;

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(win_x, win_y);
    win_id = glutCreateWindow("GPU Compute Shader Voxel Viewer");

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

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
}

void draw_objects(void)
{
    // Draw surface points using optimized function
    draw_points_fast();

    // Draw triangles using optimized function
    if (draw_triangles_on_screen) {
        draw_triangles_fast(vo.model_matrix);
    }

    // Draw axes using optimized function
    if (draw_axis) {
        draw_axis_fast();
    }
}

void display_func(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    draw_objects();
    glFlush();

    if (false == screenshot_mode)
        glutSwapBuffers();
}

void keyboard_func(unsigned char key, int x, int y)
{
    switch (tolower(key))
    {
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

    case 'o':
    {
        vo.u += 0.1f;
        vo.model_matrix = glm::mat4(1.0f);
        vo.model_matrix = glm::rotate(vo.model_matrix, vo.u, glm::vec3(0.0f, 1.0f, 0.0f));
        vo.model_matrix = glm::rotate(vo.model_matrix, vo.v, glm::vec3(1.0f, 0.0f, 0.0f));

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        get_background_points_GPU(vo);
        glFinish();
        updateSurfacePointsForRendering(vo);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> elapsed = end - start;
        cout << "GPU compute time: " << elapsed.count() << " ms" << endl;
        break;
    }
    case 'p':
    {
        vo.u -= 0.1f;
        vo.model_matrix = glm::mat4(1.0f);
        vo.model_matrix = glm::rotate(vo.model_matrix, vo.u, glm::vec3(0.0f, 1.0f, 0.0f));
        vo.model_matrix = glm::rotate(vo.model_matrix, vo.v, glm::vec3(1.0f, 0.0f, 0.0f));

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        get_background_points_GPU(vo);
        glFinish();
        updateSurfacePointsForRendering(vo);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> elapsed = end - start;
        cout << "GPU compute time: " << elapsed.count() << " ms" << endl;
        break;
    }
    case 'k':
    {
        vo.v += 0.1f;
        vo.model_matrix = glm::mat4(1.0f);
        vo.model_matrix = glm::rotate(vo.model_matrix, vo.u, glm::vec3(0.0f, 1.0f, 0.0f));
        vo.model_matrix = glm::rotate(vo.model_matrix, vo.v, glm::vec3(1.0f, 0.0f, 0.0f));

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        get_background_points_GPU(vo);
        glFinish();
        updateSurfacePointsForRendering(vo);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> elapsed = end - start;
        cout << "GPU compute time: " << elapsed.count() << " ms" << endl;
        break;
    }
    case 'l':
    {
        vo.v -= 0.1f;
        vo.model_matrix = glm::mat4(1.0f);
        vo.model_matrix = glm::rotate(vo.model_matrix, vo.u, glm::vec3(0.0f, 1.0f, 0.0f));
        vo.model_matrix = glm::rotate(vo.model_matrix, vo.v, glm::vec3(1.0f, 0.0f, 0.0f));

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        get_background_points_GPU(vo);
        glFinish();
        updateSurfacePointsForRendering(vo);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> elapsed = end - start;
        cout << "GPU compute time: " << elapsed.count() << " ms" << endl;
        break;
    }

    // Real-time continuous rotation mode
    case 'r':
    {
        static bool realtime_mode = false;
        realtime_mode = !realtime_mode;
        cout << "Real-time mode: " << (realtime_mode ? "ON" : "OFF") << endl;

        if (realtime_mode) {
            // Set up timer for continuous updates
            glutTimerFunc(16, [](int) {
                vo.u += 0.02f;
                vo.model_matrix = glm::mat4(1.0f);
                vo.model_matrix = glm::rotate(vo.model_matrix, vo.u, glm::vec3(0.0f, 1.0f, 0.0f));
                vo.model_matrix = glm::rotate(vo.model_matrix, vo.v, glm::vec3(1.0f, 0.0f, 0.0f));

                get_background_points_GPU(vo);
                updateSurfacePointsForRendering(vo);

                glutPostRedisplay();
                glutTimerFunc(16, nullptr, 0); // Continue timer
                }, 0);
        }
        break;
    }

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

    glutDestroyWindow(win_id);
}