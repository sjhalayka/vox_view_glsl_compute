#ifndef main_H
#define main_H

#include <GL/glew.h>
#pragma comment(lib, "glew32")

#include "uv_camera.h"
#include "custom_math.h"
#include "ogt_vox.h"



#include "shader_utils.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include <cstdlib>
#include <GL/glut.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <set>
#include <map>
#include <utility>
#include <ios>
#include <chrono>
using namespace std;



// OpenGL 4 additions
struct Vertex {
	float position[3];
	float color[3];
};

// Function prototypes
void idle_func(void);
void init_opengl(const int& width, const int& height);
void reshape_func(int width, int height);
void display_func(void);
void keyboard_func(unsigned char key, int x, int y);
void mouse_func(int button, int state, int x, int y);
void motion_func(int x, int y);
void passive_motion_func(int x, int y);

void draw_objects(void);
void cleanup(void);


// Global variables
custom_math::vector_3 background_colour(0.5f, 0.5f, 0.5f);
custom_math::vector_3 control_list_colour(1.0f, 1.0f, 1.0f);

bool draw_axis = false;
bool draw_control_list = true;
bool draw_triangles_on_screen = true;
uv_camera main_camera;

GLint win_id = 0;
GLint win_x = 800, win_y = 600;
float camera_w = 10;

float camera_fov = 45;
float camera_x_transform = 0;
float camera_y_transform = 0;
float u_spacer = 0.01f;
float v_spacer = 0.5f * u_spacer;
float w_spacer = 0.1f;
float camera_near = 0.01f;
float camera_far = 100.0f;

bool lmb_down = false;
bool mmb_down = false;
bool rmb_down = false;
int mouse_x = 0;
int mouse_y = 0;




const size_t x_res = 128;
const size_t y_res = 128;
const size_t z_res = 128;

const float x_grid_max = 10;
const float y_grid_max = 10;
const float z_grid_max = 10;


// ============================================================================
// MULTIPLE VOXEL OBJECTS SUPPORT
// ============================================================================

// Structure to hold voxel file info and location
struct VoxelFileInfo {
	std::string filename;
	glm::vec3 location;
};

// List of voxel files to load with their locations
// Add or remove entries here to change which voxel files are loaded
std::vector<VoxelFileInfo> voxelFiles = {
	{ "chr_knight.vox", glm::vec3(5, 0, 0) },
	{ "chr_cat.vox", glm::vec3(-5, 0, 0) }
};

// Per-object GPU buffer handles
struct VoxelObjectGPUData {
	GLuint voxelCentresSSBO = 0;
	GLuint voxelDensitiesSSBO = 0;
	GLuint gridMinMaxSSBO = 0;
	GLuint voGridCellsSSBO = 0;
};

std::vector<VoxelObjectGPUData> voxelObjectGPUData;




// Fluid simulation parameters
struct FluidParams {
	float dt = 1.0f/30.0f;              // Time step (~60 fps)
	float viscosity = 0.0001f;      // Kinematic viscosity
	float diffusion = 0.0001f;      // Density diffusion rate
	int jacobiIterations = 20;      // Pressure solver iterations

	float densityAmount = 10.0f;   // Amount of density to inject
	float velocityAmount = 100.0f;   // Amount of velocity to inject
	float densityDissipation = 0.99f; // Density dissipation per frame
	float velocityDissipation = 0.99f; // Velocity dissipation per frame

	// ============================================================================
	// NEW: Temperature, Buoyancy, and Gravity Parameters
	// ============================================================================
	float ambientTemperature = 0.0f;      // Background/ambient temperature
	float temperatureAmount = 100.0f;      // Temperature injection amount when adding source
	float temperatureDissipation = 0.99f; // Temperature dissipation per frame (cooling)

	// Buoyancy parameters (Boussinesq approximation)
	// F_buoy = (-buoyancyAlpha * density + buoyancyBeta * (T - T_ambient)) * up
	float buoyancyAlpha = 0.05f;          // Density buoyancy factor (dense fluid sinks)
	float buoyancyBeta = 1.0f;            // Temperature buoyancy factor (hot fluid rises)

	// Gravity
	float gravity = 9.81f;                // Gravity magnitude (applied in -Y direction)
	glm::vec3 gravityDirection = glm::vec3(0.0f, -1.0f, 0.0f); // Gravity direction (default: down)

	// Feature toggles
	bool enableGravity = false;           // Toggle gravity on/off (off by default for smoke)
	bool enableBuoyancy = true;           // Toggle buoyancy on/off
	bool enableTemperature = true;        // Toggle temperature field

	// Visualization
	bool visualizeTemperature = false;    // Show temperature instead of density
};

FluidParams fluidParams;
bool fluidSimEnabled = true;
bool fluidInitialized = false;

// Fluid SSBOs
GLuint velocitySSBO[2] = { 0, 0 };      // Double buffered velocity (vec4: vx, vy, vz, 0)
GLuint densitySSBO[2] = { 0, 0 };       // Double buffered density
GLuint pressureSSBO[2] = { 0, 0 };      // Double buffered pressure
GLuint divergenceSSBO = 0;             // Divergence field
GLuint obstacleSSBO = 0;               // Obstacle mask (1 = obstacle, 0 = fluid)
GLuint turbulentViscositySSBO = 0;     // Smagorinsky turbulent viscosity

// ============================================================================
// NEW: Temperature SSBO
// ============================================================================
GLuint temperatureSSBO[2] = { 0, 0 };  // Double buffered temperature field

// Fluid compute shader programs
GLuint advectionProgram = 0;
GLuint diffusionProgram = 0;
GLuint divergenceProgram = 0;
GLuint pressureProgram = 0;
GLuint gradientSubtractProgram = 0;
GLuint boundaryProgram = 0;
GLuint addSourceProgram = 0;
GLuint turbulenceProgram = 0;
GLuint obstacleProgram = 0;
GLuint visualizeProgram = 0;

// ============================================================================
// NEW: Buoyancy compute shader program
// ============================================================================
GLuint buoyancyProgram = 0;

int currentBuffer = 0;  // For double buffering

// Mouse interaction
bool injectDensity = false;
bool injectVelocity = false;
glm::vec3 lastMouseWorldPos(0.0f);
glm::vec3 currentMouseWorldPos(0.0f);
glm::vec3 mouseVelocity(0.0f);
int injectRadius = 3;

// Fluid visualization
GLuint fluidVAO = 0, fluidVBO = 0;
GLuint fluidRenderProgram = 0;
size_t numFluidPoints = 0;
float densityThreshold = 0.01f;

GLuint densityTexture = 0;
GLuint temperatureTexture = 0;
GLuint obstacleTexture = 0;  // Optional: for better boundary handling



// Blackening timing parameters
float blackenDuration = 1.0f;  // Seconds from white to black
std::chrono::steady_clock::time_point simulationStartTime;
bool simulationTimeInitialized = false;

// Helper to get elapsed time in seconds
float getElapsedSeconds() {
	if (!simulationTimeInitialized) {
		simulationStartTime = std::chrono::steady_clock::now();
		simulationTimeInitialized = true;
	}
	auto now = std::chrono::steady_clock::now();
	return std::chrono::duration<float>(now - simulationStartTime).count();
}






void updateFluidTextures()
{
	if (!fluidInitialized) return;

	int current = currentBuffer;  // or whichever buffer has latest data

	// Update density texture
	glActiveTexture(GL_TEXTURE0);
	if (densityTexture == 0) {
		glGenTextures(1, &densityTexture);
		glBindTexture(GL_TEXTURE_3D, densityTexture);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, x_res, y_res, z_res, 0, GL_RED, GL_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	}
	glBindTexture(GL_TEXTURE_3D, densityTexture);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, densitySSBO[current]);
	void* ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, x_res * y_res * z_res * sizeof(float),
		GL_MAP_READ_BIT);
	glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, x_res, y_res, z_res, GL_RED, GL_FLOAT, ptr);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	// Update temperature texture
	if (temperatureTexture == 0) {
		glGenTextures(1, &temperatureTexture);
		glBindTexture(GL_TEXTURE_3D, temperatureTexture);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, x_res, y_res, z_res, 0, GL_RED, GL_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	}
	glBindTexture(GL_TEXTURE_3D, temperatureTexture);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, temperatureSSBO[current]);
	ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, x_res * y_res * z_res * sizeof(float),
		GL_MAP_READ_BIT);
	glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, x_res, y_res, z_res, GL_RED, GL_FLOAT, ptr);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	// Optional: obstacle texture from obstacleSSBO
	if (obstacleTexture == 0) {
		glGenTextures(1, &obstacleTexture);
		glBindTexture(GL_TEXTURE_3D, obstacleTexture);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, x_res, y_res, z_res, 0, GL_RED, GL_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
	glBindTexture(GL_TEXTURE_3D, obstacleTexture);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, obstacleSSBO);
	ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, x_res * y_res * z_res * sizeof(float), GL_MAP_READ_BIT);
	glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, x_res, y_res, z_res, GL_RED, GL_FLOAT, ptr);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}


GLuint fullscreenVAO = 0, fullscreenVBO = 0;


void initFullscreenQuad()
{
	float quadVertices[] = {
		-1.0f, -1.0f, 0.0f, 0.0f,
		 1.0f, -1.0f, 1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f, 1.0f,
		 1.0f,  1.0f, 1.0f, 1.0f
	};

	glGenVertexArrays(1, &fullscreenVAO);
	glGenBuffers(1, &fullscreenVBO);
	glBindVertexArray(fullscreenVAO);
	glBindBuffer(GL_ARRAY_BUFFER, fullscreenVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glBindVertexArray(0);
}

GLuint volumeRenderProgram = 0;





// Vertex structure for shader compatibility
struct RenderVertex {
	float position[3];
	float normal[3];
	float color[3];
};

// ============================================================================
// LIGHTING STRUCTURES
// ============================================================================

const int MAX_POINT_LIGHTS = 8;
const int MAX_SPOT_LIGHTS = 8;
const int MAX_DIR_LIGHTS = 4;

struct PointLight {
	glm::vec3 position;
	glm::vec3 color;
	float intensity;
	float constant;     // Attenuation: 1.0
	float linear;       // Attenuation: 0.09
	float quadratic;    // Attenuation: 0.032
	bool enabled;

	PointLight() : position(0.0f), color(1.0f), intensity(1.0f),
		constant(1.0f), linear(0.09f), quadratic(0.032f), enabled(false) {
	}
};

struct SpotLight {
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 color;
	float intensity;
	float cutOff;       // Inner cone angle (cosine)
	float outerCutOff;  // Outer cone angle (cosine)
	float constant;
	float linear;
	float quadratic;
	bool enabled;

	SpotLight() : position(0.0f), direction(0.0f, -1.0f, 0.0f), color(1.0f),
		intensity(1.0f), cutOff(glm::cos(glm::radians(12.5f))),
		outerCutOff(glm::cos(glm::radians(17.5f))),
		constant(1.0f), linear(0.09f), quadratic(0.032f), enabled(false) {
	}
};

struct DirectionalLight {
	glm::vec3 direction;
	glm::vec3 color;
	float intensity;
	bool enabled;

	DirectionalLight() : direction(0.0f, -1.0f, 0.0f), color(1.0f),
		intensity(1.0f), enabled(false) {
	}
};

// Global light arrays
std::vector<PointLight> pointLights(MAX_POINT_LIGHTS);
std::vector<SpotLight> spotLights(MAX_SPOT_LIGHTS);
std::vector<DirectionalLight> dirLights(MAX_DIR_LIGHTS);

// Material properties
struct Material {
	float ambient;
	float shininess;
	Material() : ambient(0.1f), shininess(32.0f) {}
};

Material globalMaterial;

// Initialize default lights
void initDefaultLights() {
	// One directional light (sun-like)
	//dirLights[0].direction = glm::normalize(glm::vec3(-10.0f, -10.0f, -10.0f));
	//dirLights[0].color = glm::vec3(1.0f, 0.98f, 0.95f);
	//dirLights[0].intensity = 0.8f;
	//dirLights[0].enabled = true;

	// One point light
	//pointLights[0].position = glm::vec3(20.0f, 20.0f, 20.0f);
	//pointLights[0].color = glm::vec3(1.0f, 0.9f, 0.8f);
	//pointLights[0].intensity = 50.0f;
	//pointLights[0].enabled = true;

	spotLights[0].position = glm::vec3(20.0f, 20.0f, 20.0f);
	spotLights[0].direction = glm::normalize(glm::vec3(-10.0f, -10.0f, -10.0f));
	spotLights[0].color = glm::vec3(1.0f, 0.9f, 0.8f);
	spotLights[0].intensity = 50.0f;
	spotLights[0].enabled = true;
}


// ============================================================================
// SHADOW MAPPING STRUCTURES AND GLOBALS
// ============================================================================

// Shadow map resolution (higher = better quality, more memory)
const int SHADOW_MAP_SIZE = 2048;
const int POINT_SHADOW_MAP_SIZE = 1024;  // Cube maps are more expensive

// Shadow map arrays for each light type
GLuint dirLightShadowMaps[MAX_DIR_LIGHTS] = { 0 };
GLuint dirLightShadowFBOs[MAX_DIR_LIGHTS] = { 0 };
glm::mat4 dirLightSpaceMatrices[MAX_DIR_LIGHTS];

GLuint spotLightShadowMaps[MAX_SPOT_LIGHTS] = { 0 };
GLuint spotLightShadowFBOs[MAX_SPOT_LIGHTS] = { 0 };
glm::mat4 spotLightSpaceMatrices[MAX_SPOT_LIGHTS];

// Point lights use cube maps for omnidirectional shadows
GLuint pointLightShadowCubeMaps[MAX_POINT_LIGHTS] = { 0 };
GLuint pointLightShadowFBOs[MAX_POINT_LIGHTS] = { 0 };
// 6 view matrices per point light (one for each cube face)
glm::mat4 pointLightShadowMatrices[MAX_POINT_LIGHTS * 6];
float pointLightFarPlanes[MAX_POINT_LIGHTS] = { 25.0f }; // Adjustable per light

// Shadow rendering shader programs
GLuint shadowDepthProgram = 0;           // For directional and spot lights
GLuint pointShadowDepthProgram = 0;      // For point light cube maps

// Shadow mapping parameters
struct ShadowParams {
	float bias = 0.0f;              // Depth bias to prevent shadow acne
	float normalBias = 0.02f;         // Normal-based bias
	int pcfSamples = 1;               // PCF kernel size (2 = 5x5 samples)
	bool enableShadows = true;        // Global shadow toggle
	float shadowIntensity = 0.5f;     // 0 = full shadow, 1 = no shadow effect
};

ShadowParams shadowParams;

// Function declarations
void initShadowMaps();
void cleanupShadowMaps();
void renderShadowMaps();
void renderDirLightShadowMap(int lightIndex);
void renderSpotLightShadowMap(int lightIndex);
void renderPointLightShadowMap(int lightIndex);
void setShadowUniforms(GLuint program);






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






// Add this function to main.cpp after the fluid simulation functions
float getFluidDensityAtPoint(const glm::vec3& worldPos) {
	if (!fluidInitialized) return 0.0f;

	glm::vec3 bgGridMin(-x_grid_max, -y_grid_max, -z_grid_max);
	glm::vec3 bgGridMax(x_grid_max, y_grid_max, z_grid_max);
	glm::vec3 gridSize = glm::vec3(x_res, y_res, z_res);
	glm::vec3 cellSize = (bgGridMax - bgGridMin) / (gridSize - 1.0f);

	// Convert world position to grid coordinates
	glm::vec3 gridPos = (worldPos - bgGridMin) / cellSize;
	gridPos = glm::clamp(gridPos, glm::vec3(0.5f), gridSize - glm::vec3(1.5f));

	ivec3 cell = glm::ivec3(glm::floor(gridPos));

	// Ensure within bounds
	cell.x = glm::clamp(cell.x, 0, int(x_res - 1));
	cell.y = glm::clamp(cell.y, 0, int(y_res - 1));
	cell.z = glm::clamp(cell.z, 0, int(z_res - 1));

	size_t index = cell.x + cell.y * x_res + cell.z * x_res * y_res;

	// Read density at this position
	float density = 0.0f;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, densitySSBO[0]);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
		index * sizeof(float),
		sizeof(float),
		&density);

	return density;
}


class voxel_object
{
public:



	custom_math::vertex_3 vo_grid_min;
	custom_math::vertex_3 vo_grid_max;
	// 3D array of voxel indices (stores -1 for empty cells)

	const float cell_size = 1.0;

	// Triangles data
	vector<custom_math::triangle> tri_vec;
	//custom_math::vertex_3 min_location, max_location;

	vector<glm::ivec3> voxel_indices;
	vector<custom_math::vertex_3> voxel_centres;

	// Note: when destroying a voxel, set voxel_densities[index] to 0 and vo_grid_cells[index] to -1
	// then re-generate the triangles
	vector<float> voxel_densities;
	std::vector<long long signed int> vo_grid_cells;



	vector<glm::vec4> voxel_colours;
	vector<glm::vec4> voxel_original_colours;  // Store original colors
	vector<float> voxel_blacken_times;

	size_t voxel_x_res;
	size_t voxel_y_res;
	size_t voxel_z_res;

	vector<glm::ivec3> background_indices;
	vector<custom_math::vertex_3> background_centres;
	vector<float> background_densities;
	vector<size_t> background_collisions;

	vector<glm::ivec3> background_surface_indices;
	vector<custom_math::vertex_3> background_surface_centres;
	vector<float> background_surface_densities;
	vector<vector<size_t>> background_surface_collisions;

	glm::mat4 model_matrix = glm::mat4(1.0f);
	float u = 0.0f, v = 0.0f;




	//// Initialize the grid based on voxel data
	//void initialize(const std::vector<custom_math::vertex_3>& voxel_centres/*,
	//	const std::vector<float>& voxel_densities*/) {

	//	// Find min/max extents
	//	if (voxel_centres.empty()) return;


	//}



	// Find which voxel contains a point
	bool find_voxel_containing_point(
		const custom_math::vertex_3& point,
		size_t& voxel_index) const
	{
		// Get grid cell coordinates
		float cellSize = cell_size;

		int cell_x = static_cast<int>(floor((point.x - vo_grid_min.x) / cellSize));
		int cell_y = static_cast<int>(floor((point.y - vo_grid_min.y) / cellSize));
		int cell_z = static_cast<int>(floor((point.z - vo_grid_min.z) / cellSize));

		// Check bounds
		if (cell_x < 0 || cell_x >= static_cast<int>(voxel_x_res) ||
			cell_y < 0 || cell_y >= static_cast<int>(voxel_y_res) ||
			cell_z < 0 || cell_z >= static_cast<int>(voxel_z_res)) {
			return false;
		}

		// Find the index in the flattened 3D array
		size_t cellIndex = cell_x + cell_y * voxel_x_res + cell_z * voxel_x_res * voxel_y_res;

		if (cellIndex >= vo_grid_cells.size()) {
			return false;
		}

		// Use the grid cells lookup to get actual voxel index
		long long voxelIdx = vo_grid_cells[cellIndex];

		if (voxelIdx == -1) {
			return false;
		}

		// Precise check against the voxel
		float halfSize = cellSize * 0.5f;
		const custom_math::vertex_3& center = voxel_centres[voxelIdx];

		if (point.x >= center.x - halfSize && point.x <= center.x + halfSize &&
			point.y >= center.y - halfSize && point.y <= center.y + halfSize &&
			point.z >= center.z - halfSize && point.z <= center.z + halfSize) {
			voxel_index = static_cast<size_t>(voxelIdx);
			return true;
		}

		return false;
	}
};

std::vector<voxel_object> voxel_objects;


void get_surface_points_GPU(voxel_object& v);


// Define the coordinates for 6 adjacent neighbors (up, down, left, right, front, back)
static const int directions[6][3] = {
	{1, 0, 0}, {-1, 0, 0},  // x directions
	{0, 1, 0}, {0, -1, 0},  // y directions
	{0, 0, 1}, {0, 0, -1}   // z directions
};

// Initialize with the same grid size as background points
//const size_t x_res = background_indices.empty() ? 0 : background_indices[background_indices.size() - 1].x + 1;
//const size_t y_res = background_indices.empty() ? 0 : background_indices[background_indices.size() - 1].y + 1;
//const size_t z_res = background_indices.empty() ? 0 : background_indices[background_indices.size() - 1].z + 1;

// Clear any existing data


void get_surface_points(void)
{


	//	std::cout << "Found " << background_surface_centres.size() << " surface points" << std::endl;
}


bool gpuInitialized = false;
size_t numSurfacePoints = 0;

// ============================================================================
// Update triangle buffer for rendering
// ============================================================================

size_t numTriangleIndices = 0;

void updateTriangleBuffer(std::vector<voxel_object>& objects) {
	if (!gpuInitialized) return;

	vector<RenderVertex> vertices;
	vector<GLuint> indices;

	// Combine triangles from all voxel objects
	for (auto& v : objects) {
		for (size_t i = 0; i < v.tri_vec.size(); i++) {
			// Compute face normal from triangle vertices
			glm::vec3 v0(v.tri_vec[i].vertex[0].x, v.tri_vec[i].vertex[0].y, v.tri_vec[i].vertex[0].z);
			glm::vec3 v1(v.tri_vec[i].vertex[1].x, v.tri_vec[i].vertex[1].y, v.tri_vec[i].vertex[1].z);
			glm::vec3 v2(v.tri_vec[i].vertex[2].x, v.tri_vec[i].vertex[2].y, v.tri_vec[i].vertex[2].z);

			glm::vec3 edge1 = v1 - v0;
			glm::vec3 edge2 = v2 - v0;
			glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

			// Transform normal by model matrix (use normal matrix for correct transformation)
			glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(v.model_matrix)));
			glm::vec3 worldNormal = glm::normalize(normalMatrix * faceNormal);

			for (size_t j = 0; j < 3; j++) {
				RenderVertex rv;

				// Transform vertices by model matrix
				glm::vec4 worldPos = v.model_matrix * glm::vec4(
					v.tri_vec[i].vertex[j].x,
					v.tri_vec[i].vertex[j].y,
					v.tri_vec[i].vertex[j].z,
					1.0f);

				rv.position[0] = worldPos.x;
				rv.position[1] = worldPos.y;
				rv.position[2] = worldPos.z;
				rv.normal[0] = worldNormal.x;
				rv.normal[1] = worldNormal.y;
				rv.normal[2] = worldNormal.z;
				rv.color[0] = v.tri_vec[i].colour.x;
				rv.color[1] = v.tri_vec[i].colour.y;
				rv.color[2] = v.tri_vec[i].colour.z;
				vertices.push_back(rv);
				indices.push_back(static_cast<GLuint>(vertices.size() - 1));
			}
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

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
	glEnableVertexAttribArray(0);
	// Normal attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// Color attribute
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);
}






// Overload for single voxel object (backwards compatibility)
void updateTriangleBuffer(voxel_object& v) {
	std::vector<voxel_object> temp = { v };
	updateTriangleBuffer(temp);
}



void centre_voxels_on_xyz(voxel_object& v)
{
	float x_min = numeric_limits<float>::max();
	float y_min = numeric_limits<float>::max();
	float z_min = numeric_limits<float>::max();
	float x_max = -numeric_limits<float>::max();
	float y_max = -numeric_limits<float>::max();
	float z_max = -numeric_limits<float>::max();

	for (size_t t = 0; t < v.voxel_centres.size(); t++)
	{
		if (v.voxel_densities[t] == 0)
			continue;

		if (v.voxel_centres[t].x < x_min)
			x_min = v.voxel_centres[t].x;

		if (v.voxel_centres[t].x > x_max)
			x_max = v.voxel_centres[t].x;

		if (v.voxel_centres[t].y < y_min)
			y_min = v.voxel_centres[t].y;

		if (v.voxel_centres[t].y > y_max)
			y_max = v.voxel_centres[t].y;

		if (v.voxel_centres[t].z < z_min)
			z_min = v.voxel_centres[t].z;

		if (v.voxel_centres[t].z > z_max)
			z_max = v.voxel_centres[t].z;
	}

	for (size_t t = 0; t < v.voxel_centres.size(); t++)
	{
		v.voxel_centres[t].x += -(x_max + x_min) / 2.0f;
		v.voxel_centres[t].y += -(y_max + y_min) / 2.0f;
		v.voxel_centres[t].z += -(z_max + z_min) / 2.0f;
	}
}

bool get_voxels(const char* file_name, voxel_object& v)
{
	v.voxel_indices.clear();
	v.voxel_centres.clear();
	v.voxel_densities.clear();
	v.voxel_colours.clear();
	v.vo_grid_cells.clear();

	ifstream infile(file_name, ifstream::ate | ifstream::binary);

	if (infile.fail())
	{
		cout << "Could not open file " << file_name << endl;
		return false;
	}

	size_t file_size = infile.tellg();

	infile.close();

	if (file_size == 0)
		return false;

	infile.open(file_name, ifstream::binary);

	if (infile.fail())
	{
		return false;
	}

	vector<unsigned char> f(file_size, 0);

	infile.read(reinterpret_cast<char*>(&f[0]), file_size);
	infile.close();

	const ogt_vox_scene* scene = ogt_vox_read_scene(&f[0], static_cast<uint32_t>(file_size));

	v.voxel_x_res = scene->models[0]->size_x;
	v.voxel_y_res = scene->models[0]->size_y;
	v.voxel_z_res = scene->models[0]->size_z;

	v.voxel_indices.resize(v.voxel_x_res * v.voxel_y_res * v.voxel_z_res);
	v.voxel_centres.resize(v.voxel_x_res * v.voxel_y_res * v.voxel_z_res);
	v.voxel_densities.resize(v.voxel_x_res * v.voxel_y_res * v.voxel_z_res);
	v.voxel_colours.resize(v.voxel_x_res * v.voxel_y_res * v.voxel_z_res);
	v.vo_grid_cells.resize(v.voxel_x_res * v.voxel_y_res * v.voxel_z_res);

	for (size_t x = 0; x < v.voxel_x_res; x++)
	{
		for (size_t y = 0; y < v.voxel_y_res; y++)
		{
			for (size_t z = 0; z < v.voxel_z_res; z++)
			{
				const size_t voxel_index = x + (y * v.voxel_x_res) + (z * v.voxel_x_res * v.voxel_y_res);
				const uint8_t colour_index = scene->models[0]->voxel_data[voxel_index];

				custom_math::vertex_3 translate(x * v.cell_size, y * v.cell_size, z * v.cell_size);

				v.voxel_centres[voxel_index] = translate;
				v.voxel_indices[voxel_index] = glm::ivec3(x, y, z);

				// Transparent
				if (colour_index == 0)
				{
					v.voxel_densities[voxel_index] = 0.0;
					v.vo_grid_cells[voxel_index] = -1;
					continue;
				}
				else
				{
					v.voxel_densities[voxel_index] = 1.0;
					v.vo_grid_cells[voxel_index] = 0;
				}

				const ogt_vox_rgba colour = scene->palette.color[colour_index];

				uint8_t r = colour.r;
				uint8_t g = colour.g;
				uint8_t b = colour.b;
				uint8_t a = colour.a;

				v.voxel_colours[voxel_index] = glm::vec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
			}
		}
	}

	ogt_vox_destroy_scene(scene);

	for (size_t i = 0; i < v.voxel_centres.size(); i++)
	{
		static const float pi = 4.0f * atanf(1.0f);
		v.voxel_centres[i].rotate_x(pi - pi / 2.0f);
	}

	centre_voxels_on_xyz(v);


	v.vo_grid_min = v.voxel_centres[0];
	v.vo_grid_max = v.voxel_centres[0];

	for (const auto& center : v.voxel_centres)
	{
		v.vo_grid_min.x = std::min(v.vo_grid_min.x, center.x - v.cell_size / 2.0f);
		v.vo_grid_min.y = std::min(v.vo_grid_min.y, center.y - v.cell_size / 2.0f);
		v.vo_grid_min.z = std::min(v.vo_grid_min.z, center.z - v.cell_size / 2.0f);

		v.vo_grid_max.x = std::max(v.vo_grid_max.x, center.x + v.cell_size / 2.0f);
		v.vo_grid_max.y = std::max(v.vo_grid_max.y, center.y + v.cell_size / 2.0f);
		v.vo_grid_max.z = std::max(v.vo_grid_max.z, center.z + v.cell_size / 2.0f);
	}

	// Calculate grid dimensions
	float size_x = v.vo_grid_max.x - v.vo_grid_min.x;
	float size_y = v.vo_grid_max.y - v.vo_grid_min.y;
	float size_z = v.vo_grid_max.z - v.vo_grid_min.z;

	// Place voxels in the grid
	for (size_t i = 0; i < v.voxel_centres.size(); i++)
	{
		if (v.voxel_densities[i] <= 0.0f) continue;

		const auto& center = v.voxel_centres[i];

		// Get grid cell coordinates
		size_t cell_x = static_cast<int>((center.x - v.vo_grid_min.x) / v.cell_size);
		size_t cell_y = static_cast<int>((center.y - v.vo_grid_min.y) / v.cell_size);
		size_t cell_z = static_cast<int>((center.z - v.vo_grid_min.z) / v.cell_size);

		// Ensure within bounds
		cell_x = std::max((size_t)0, std::min(cell_x, v.voxel_x_res - 1));
		cell_y = std::max((size_t)0, std::min(cell_y, v.voxel_y_res - 1));
		cell_z = std::max((size_t)0, std::min(cell_z, v.voxel_z_res - 1));

		// Get index in the flattened 3D array
		size_t cell_index = cell_x + (cell_y * v.voxel_x_res) + (cell_z * v.voxel_x_res * v.voxel_y_res);

		// Store voxel index in the grid
		v.vo_grid_cells[cell_index] = static_cast<int>(i);
	}

	return true;
}




bool get_triangles(vector<custom_math::triangle>& tri_vec, voxel_object& v)
{
	tri_vec.clear();

	for (size_t i = 0; i < v.voxel_centres.size(); i++)
	{
		static const float pi = 4.0f * atanf(1.0f);
		v.voxel_centres[i].rotate_x(-(pi - pi / 2.0f));
	}

	for (size_t x = 0; x < v.voxel_x_res; x++)
	{
		for (size_t y = 0; y < v.voxel_y_res; y++)
		{
			for (size_t z = 0; z < v.voxel_z_res; z++)
			{
				const size_t voxel_index = x + (y * v.voxel_x_res) + (z * v.voxel_x_res * v.voxel_y_res);
				const custom_math::vertex_3 translate = v.voxel_centres[voxel_index];

				v.voxel_indices[voxel_index] = glm::ivec3(x, y, z);

				if (0 == v.voxel_densities[voxel_index])
					continue;

				custom_math::quad q0, q1, q2, q3, q4, q5;

				// Top face (y = 1.0f)
				q0.vertex[0] = custom_math::vertex_3(v.cell_size * 0.5f, v.cell_size * 0.5f, -v.cell_size * 0.5f) + translate;
				q0.vertex[1] = custom_math::vertex_3(-v.cell_size * 0.5f, v.cell_size * 0.5f, -v.cell_size * 0.5f) + translate;
				q0.vertex[2] = custom_math::vertex_3(-v.cell_size * 0.5f, v.cell_size * 0.5f, v.cell_size * 0.5f) + translate;
				q0.vertex[3] = custom_math::vertex_3(v.cell_size * 0.5f, v.cell_size * 0.5f, v.cell_size * 0.5f) + translate;

				// Bottom face (y = -v.cell_size*0.5f)
				q1.vertex[0] = custom_math::vertex_3(v.cell_size * 0.5f, -v.cell_size * 0.5f, v.cell_size * 0.5f) + translate;
				q1.vertex[1] = custom_math::vertex_3(-v.cell_size * 0.5f, -v.cell_size * 0.5f, v.cell_size * 0.5f) + translate;
				q1.vertex[2] = custom_math::vertex_3(-v.cell_size * 0.5f, -v.cell_size * 0.5f, -v.cell_size * 0.5f) + translate;
				q1.vertex[3] = custom_math::vertex_3(v.cell_size * 0.5f, -v.cell_size * 0.5f, -v.cell_size * 0.5f) + translate;

				// Front face  (z = v.cell_size*0.5f)
				q2.vertex[0] = custom_math::vertex_3(v.cell_size * 0.5f, v.cell_size * 0.5f, v.cell_size * 0.5f) + translate;
				q2.vertex[1] = custom_math::vertex_3(-v.cell_size * 0.5f, v.cell_size * 0.5f, v.cell_size * 0.5f) + translate;
				q2.vertex[2] = custom_math::vertex_3(-v.cell_size * 0.5f, -v.cell_size * 0.5f, v.cell_size * 0.5f) + translate;
				q2.vertex[3] = custom_math::vertex_3(v.cell_size * 0.5f, -v.cell_size * 0.5f, v.cell_size * 0.5f) + translate;

				// Back face (z = -v.cell_size*0.5f)
				q3.vertex[0] = custom_math::vertex_3(v.cell_size * 0.5f, -v.cell_size * 0.5f, -v.cell_size * 0.5f) + translate;
				q3.vertex[1] = custom_math::vertex_3(-v.cell_size * 0.5f, -v.cell_size * 0.5f, -v.cell_size * 0.5f) + translate;
				q3.vertex[2] = custom_math::vertex_3(-v.cell_size * 0.5f, v.cell_size * 0.5f, -v.cell_size * 0.5f) + translate;
				q3.vertex[3] = custom_math::vertex_3(v.cell_size * 0.5f, v.cell_size * 0.5f, -v.cell_size * 0.5f) + translate;

				// Right face (x = v.cell_size*0.5f)
				q4.vertex[0] = custom_math::vertex_3(v.cell_size * 0.5f, v.cell_size * 0.5f, -v.cell_size * 0.5f) + translate;
				q4.vertex[1] = custom_math::vertex_3(v.cell_size * 0.5f, v.cell_size * 0.5f, v.cell_size * 0.5f) + translate;
				q4.vertex[2] = custom_math::vertex_3(v.cell_size * 0.5f, -v.cell_size * 0.5f, v.cell_size * 0.5f) + translate;
				q4.vertex[3] = custom_math::vertex_3(v.cell_size * 0.5f, -v.cell_size * 0.5f, -v.cell_size * 0.5f) + translate;

				// Left face (x = -v.cell_size*0.5f)
				q5.vertex[0] = custom_math::vertex_3(-v.cell_size * 0.5f, v.cell_size * 0.5f, v.cell_size * 0.5f) + translate;
				q5.vertex[1] = custom_math::vertex_3(-v.cell_size * 0.5f, v.cell_size * 0.5f, -v.cell_size * 0.5f) + translate;
				q5.vertex[2] = custom_math::vertex_3(-v.cell_size * 0.5f, -v.cell_size * 0.5f, -v.cell_size * 0.5f) + translate;
				q5.vertex[3] = custom_math::vertex_3(-v.cell_size * 0.5f, -v.cell_size * 0.5f, v.cell_size * 0.5f) + translate;

				custom_math::triangle t;

				const glm::vec4 c = v.voxel_colours[voxel_index];
				t.colour.x = c.r;
				t.colour.y = c.g;
				t.colour.z = c.b;

				size_t neighbour_index = 0;

				// Note that this index is possibly out of range, 
				// which is why it's used second in the if()
				neighbour_index = x + (y + 1) * v.voxel_x_res + z * v.voxel_x_res * v.voxel_y_res;
				if (y == v.voxel_y_res - 1 || 0 == v.voxel_densities[neighbour_index])
				{
					t.vertex[0] = q0.vertex[0];
					t.vertex[1] = q0.vertex[1];
					t.vertex[2] = q0.vertex[2];

					tri_vec.push_back(t);

					t.vertex[0] = q0.vertex[0];
					t.vertex[1] = q0.vertex[2];
					t.vertex[2] = q0.vertex[3];
					tri_vec.push_back(t);
				}

				// Note that this index is possibly out of range, 
				// which is why it's used second in the if()
				neighbour_index = x + (y - 1) * v.voxel_x_res + z * v.voxel_x_res * v.voxel_y_res;
				if (y == 0 || 0 == v.voxel_densities[neighbour_index])
				{
					t.vertex[0] = q1.vertex[0];
					t.vertex[1] = q1.vertex[1];
					t.vertex[2] = q1.vertex[2];
					tri_vec.push_back(t);

					t.vertex[0] = q1.vertex[0];
					t.vertex[1] = q1.vertex[2];
					t.vertex[2] = q1.vertex[3];
					tri_vec.push_back(t);
				}


				// Note that this index is possibly out of range, 
				// which is why it's used second in the if()
				neighbour_index = x + y * v.voxel_x_res + (z + 1) * v.voxel_x_res * v.voxel_y_res;
				if (z == v.voxel_z_res - 1 || 0 == v.voxel_densities[neighbour_index])
				{
					t.vertex[0] = q2.vertex[0];
					t.vertex[1] = q2.vertex[1];
					t.vertex[2] = q2.vertex[2];
					tri_vec.push_back(t);

					t.vertex[0] = q2.vertex[0];
					t.vertex[1] = q2.vertex[2];
					t.vertex[2] = q2.vertex[3];
					tri_vec.push_back(t);
				}


				// Note that this index is possibly out of range, 
				// which is why it's used second in the if()
				neighbour_index = x + (y)*v.voxel_x_res + (z - 1) * v.voxel_x_res * v.voxel_y_res;
				if (z == 0 || 0 == v.voxel_densities[neighbour_index])
				{
					t.vertex[0] = q3.vertex[0];
					t.vertex[1] = q3.vertex[1];
					t.vertex[2] = q3.vertex[2];
					tri_vec.push_back(t);

					t.vertex[0] = q3.vertex[0];
					t.vertex[1] = q3.vertex[2];
					t.vertex[2] = q3.vertex[3];
					tri_vec.push_back(t);
				}


				// Note that this index is possibly out of range, 
				// which is why it's used second in the if()
				neighbour_index = (x + 1) + (y)*v.voxel_x_res + (z)*v.voxel_x_res * v.voxel_y_res;
				if (x == v.voxel_x_res - 1 || 0 == v.voxel_densities[neighbour_index])
				{
					t.vertex[0] = q4.vertex[0];
					t.vertex[1] = q4.vertex[1];
					t.vertex[2] = q4.vertex[2];
					tri_vec.push_back(t);

					t.vertex[0] = q4.vertex[0];
					t.vertex[1] = q4.vertex[2];
					t.vertex[2] = q4.vertex[3];
					tri_vec.push_back(t);
				}

				// Note that this index is possibly out of range, 
				// which is why it's used second in the if()
				neighbour_index = (x - 1) + (y)*v.voxel_x_res + (z)*v.voxel_x_res * v.voxel_y_res;
				if (x == 0 || 0 == v.voxel_densities[neighbour_index])
				{
					t.vertex[0] = q5.vertex[0];
					t.vertex[1] = q5.vertex[1];
					t.vertex[2] = q5.vertex[2];
					tri_vec.push_back(t);

					t.vertex[0] = q5.vertex[0];
					t.vertex[1] = q5.vertex[2];
					t.vertex[2] = q5.vertex[3];
					tri_vec.push_back(t);
				}
			}
		}
	}



	cout << tri_vec.size() << endl;







	for (size_t i = 0; i < tri_vec.size(); i++)
	{
		static const float pi = 4.0f * atanf(1.0f);
		tri_vec[i].vertex[0].rotate_x(pi - pi / 2.0f);
		tri_vec[i].vertex[1].rotate_x(pi - pi / 2.0f);
		tri_vec[i].vertex[2].rotate_x(pi - pi / 2.0f);
	}

	for (size_t i = 0; i < v.voxel_centres.size(); i++)
	{
		static const float pi = 4.0f * atanf(1.0f);
		v.voxel_centres[i].rotate_x(pi - pi / 2.0f);
	}


	return true;
}





glm::vec4 getBlackenColor(float t, const glm::vec4& originalColor) {
	// t goes from 0 (just blackened) to 1 (fully black)
	t = glm::clamp(t, 0.0f, 1.0f);

	// Color keyframes: white -> yellow -> orange -> red -> black
	// We blend from original color through these stages
	glm::vec4 white(1.0f, 1.0f, 1.0f, 1.0f);
	glm::vec4 yellow(1.0f, 1.0f, 0.0f, 1.0f);
	glm::vec4 orange(1.0f, 0.5f, 0.0f, 1.0f);
	glm::vec4 red(1.0f, 0.0f, 0.0f, 1.0f);
	glm::vec4 black(0.0f, 0.0f, 0.0f, 1.0f);

	glm::vec4 targetColor;
	if (t < 0.25f) {
		// white -> yellow
		float localT = t / 0.25f;
		targetColor = glm::mix(white, yellow, localT);
	}
	else if (t < 0.5f) {
		// yellow -> orange
		float localT = (t - 0.25f) / 0.25f;
		targetColor = glm::mix(yellow, orange, localT);
	}
	else if (t < 0.75f) {
		// orange -> red
		float localT = (t - 0.5f) / 0.25f;
		targetColor = glm::mix(orange, red, localT);
	}
	else {
		// red -> black
		float localT = (t - 0.75f) / 0.25f;
		targetColor = glm::mix(red, black, localT);
	}

	return targetColor;
}

void updateBlackenColors(voxel_object& v) {
	bool anyChanged = false;
	float currentSeconds = getElapsedSeconds();

	for (size_t i = 0; i < v.voxel_blacken_times.size(); i++) {
		if (v.voxel_blacken_times[i] >= 0.0f) {
			float elapsed = currentSeconds - v.voxel_blacken_times[i];
			float t = elapsed / blackenDuration;

			glm::vec4 newColor = getBlackenColor(t, v.voxel_original_colours[i]);

			if (v.voxel_colours[i] != newColor) {
				v.voxel_colours[i] = newColor;
				anyChanged = true;
			}
		}
	}

	if (anyChanged) {
		v.tri_vec.clear();
		get_triangles(v.tri_vec, v);
		updateTriangleBuffer(voxel_objects);
	}
}



// Mark voxels for blackening (records timestamp, doesn't change color immediately)
void do_blackening(voxel_object& v)
{
	if (!gpuInitialized) {
		cout << "GPU not initialized!" << endl;
		return;
	}

	// Initialize blacken_times if needed
	if (v.voxel_blacken_times.empty()) {
		v.voxel_blacken_times.resize(v.voxel_centres.size(), -1.0f);
	}

	// Store original colors if not already stored
	if (v.voxel_original_colours.empty()) {
		v.voxel_original_colours = v.voxel_colours;
	}

	// Read all necessary data from GPU
	size_t gridSize = x_res * y_res * z_res;

	// Read surface densities
	vector<float> surfaceDensities(gridSize);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, surfaceDensitiesSSBO);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
		gridSize * sizeof(float),
		surfaceDensities.data());

	// Read background densities (1.0 = inside voxel, 0.0 = outside)
	vector<float> backgroundDensities(gridSize);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, backgroundDensitiesSSBO);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
		gridSize * sizeof(float),
		backgroundDensities.data());

	// Read background collisions (voxel indices for points inside voxels)
	vector<int> backgroundCollisions(gridSize);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, backgroundCollisionsSSBO);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
		gridSize * sizeof(int),
		backgroundCollisions.data());

	// Read fluid densities if available
	vector<float> fluidDensities;
	if (fluidInitialized && fluidSimEnabled) {
		fluidDensities.resize(gridSize);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, densitySSBO[0]);
		glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
			gridSize * sizeof(float),
			fluidDensities.data());
	}

	// Check all 6 neighbor directions
	const ivec3 dirs[6] = {
		ivec3(1, 0, 0), ivec3(-1, 0, 0),
		ivec3(0, 1, 0), ivec3(0, -1, 0),
		ivec3(0, 0, 1), ivec3(0, 0, -1)
	};

	int newlyBlackenedCount = 0;

	// Process each grid point
	for (size_t x = 0; x < x_res; x++)
	{
		for (size_t y = 0; y < y_res; y++)
		{
			for (size_t z = 0; z < z_res; z++)
			{
				const size_t index = x + y * x_res + z * x_res * y_res;

				// Check if this is a surface point
				if (surfaceDensities[index] <= 0.0f)
					continue;

				// Check if this surface point has fluid density
				bool hasFluid = false;
				if (!fluidDensities.empty()) {
					float fluidDensity = fluidDensities[index];
					if (fluidDensity > 0.1f) {
						hasFluid = true;
					}
				}

				// If surface point has fluid, check its neighbors for voxels
				if (hasFluid) {
					ivec3 gid(x, y, z);

					for (int d = 0; d < 6; d++) {
						ivec3 neighbor = gid + dirs[d];

						// Check bounds
						if (neighbor.x < 0 || neighbor.x >= x_res ||
							neighbor.y < 0 || neighbor.y >= y_res ||
							neighbor.z < 0 || neighbor.z >= z_res) {
							continue;
						}

						size_t neighborIndex = neighbor.x + neighbor.y * x_res + neighbor.z * x_res * y_res;

						// If neighbor is inside a voxel, get the voxel index
						if (backgroundDensities[neighborIndex] > 0.0f) {
							int voxelIndex = backgroundCollisions[neighborIndex];
							if (voxelIndex >= 0 && voxelIndex < static_cast<int>(v.voxel_centres.size())) {
								// Only mark if not already blackening
								if (1 /*v.voxel_blacken_times[voxelIndex] < 0.0f*/) {
									v.voxel_blacken_times[voxelIndex] = getElapsedSeconds();
									newlyBlackenedCount++;
								}
							}
						}
					}
				}
			}
		}
	}

	// Update colors based on all blackening timestamps
	updateBlackenColors(v);

	if (newlyBlackenedCount > 0) {
		cout << "Newly blackened " << newlyBlackenedCount << " voxels" << endl;
	}
}



const int edgeTable[256] = {
	0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

// Triangle table - for each of 256 configurations, up to 5 triangles (15 edge indices, -1 terminated)
const int triTable[256][16] = {
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};




// ============================================================================
// MARCHING CUBES - Global Variables
// ============================================================================

// Number of isosurface layers to generate
const int NUM_ISO_LAYERS = 4;

// Isovalue thresholds for each layer (adjust based on your density range)
float isoValues[NUM_ISO_LAYERS] = { 0.1f, 0.3f, 0.5f, 0.8f };

// Opacity for each layer (outer layers more transparent)
float isoOpacities[NUM_ISO_LAYERS] = { 0.1f, 0.7f, 0.9f, 1.0f };// { 0.15f, 0.25f, 0.4f, 0.7f };

// Colors for each layer (can be gradient from outer to inner)
glm::vec4 isoColors[NUM_ISO_LAYERS] = {
	glm::vec4(0.2f, 0.5f, 0.9f, isoOpacities[0]),  // Outer: light blue, very transparent
	glm::vec4(0.0, 0.125, 0.25f, isoOpacities[1]),  // 
	glm::vec4(1.0f, 0.0f, 0.0f, isoOpacities[2]),  //
	glm::vec4(0.2f, 0.0f, 0.0f, isoOpacities[3])    // Inner: white-ish, more opaque
};

// Marching Cubes SSBOs
GLuint mcEdgeTableSSBO = 0;
GLuint mcTriTableSSBO = 0;
GLuint mcVertexCountSSBO = 0;        // Atomic counter for vertices
GLuint mcVertexBufferSSBO = 0;       // Output vertices
GLuint mcNormalBufferSSBO = 0;       // Output normals

// Per-layer vertex counts and offsets
GLuint mcLayerVertexCounts[NUM_ISO_LAYERS] = { 0 };
GLuint mcLayerVertexOffsets[NUM_ISO_LAYERS] = { 0 };

// Marching Cubes compute shader program
GLuint mcComputeProgram = 0;
GLuint mcCountProgram = 0;  // For counting vertices first

// Marching Cubes render program (with transparency)
GLuint mcRenderProgram = 0;

// VAO for rendering marching cubes output
GLuint mcVAO = 0;
GLuint mcVBO = 0;
GLuint mcNormalVBO = 0;

const size_t MC_MAX_VERTICES = x_res * y_res * z_res * 5 * 3; // Max 5 triangles per cell

// Toggle for marching cubes vs ray marching
bool useMarchingCubes = false;

// ============================================================================
// MARCHING CUBES - Function Declarations
// ============================================================================

void initMarchingCubes();
void runMarchingCubes(float isoValue, int layerIndex);
void runAllMarchingCubesLayers();
void drawMarchingCubesMesh();
void cleanupMarchingCubes();








// Utility functions
void checkGLError(const char* operation) {
	GLenum error = glGetError();
	if (error != GL_NO_ERROR) {
		std::cerr << "OpenGL error after " << operation << ": " << error << std::endl;
	}
}

GLuint compileShader(GLenum type, const char* source) {
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, nullptr);
	glCompileShader(shader);

	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetShaderInfoLog(shader, 512, nullptr, infoLog);
		std::cerr << "Shader compilation failed:\n" << infoLog << std::endl;
		std::cerr << "Source:\n" << source << std::endl;
	}

	return shader;
}



const char* textVertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec4 aColor;

uniform mat4 projection;
uniform mat4 model;

out vec2 TexCoord;
out vec4 Color;

void main() {
    gl_Position = projection * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    Color = aColor;
}
)";

const char* textFragmentShaderSource = R"(
#version 330 core
in vec2 TexCoord;
in vec4 Color;

uniform sampler2D fontTexture;
uniform bool useColor;

out vec4 FragColor;

void main() {
    vec4 texColor = texture(fontTexture, TexCoord);
    
    // Handle different font texture formats
    if (useColor) {
        // For colored font atlas, just blend with the vertex color
        FragColor = texColor * Color;
    } else {
        // For grayscale/alpha font atlas, use the alpha/red channel
        // with the vertex color
        float alpha = texColor.r; // or texColor.a depending on your font texture
        FragColor = vec4(Color.rgb, Color.a * alpha);
    }
}
)";






struct FontAtlas {
	GLuint textureID;
	int charWidth;      // Width of each character (16)
	int charHeight;     // Height of each character (16)
	int atlasWidth;     // Atlas width (256)
	int atlasHeight;    // Atlas height (256)
	int charsPerRow;    // Characters per row in the atlas (16)
};








#endif