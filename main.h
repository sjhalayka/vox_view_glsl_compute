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

bool draw_axis = true;
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


vector<unsigned char> test_texture;

const size_t x_res = 128;
const size_t y_res = 128;
const size_t z_res = 128;

const float x_grid_max = 10;
const float y_grid_max = 10;
const float z_grid_max = 10;

glm::vec3 knight_location = glm::vec3(10, 0, 0);
glm::vec3 cat_location = glm::vec3(-10, 0, 0);


// Fluid simulation parameters
struct FluidParams {
	float dt = 0.016f;              // Time step (~60 fps)
	float viscosity = 0.0001f;      // Kinematic viscosity
	float diffusion = 0.0001f;      // Density diffusion rate
	int jacobiIterations = 40;      // Pressure solver iterations
	float smagorinskyConst = 0.1f;  // Smagorinsky constant for LES turbulence
	float densityAmount = 10.0f;   // Amount of density to inject
	float velocityAmount = 10.0f;   // Amount of velocity to inject
	float densityDissipation = 0.995f; // Density dissipation per frame
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
	bool visualizeTemperature = true;    // Show temperature instead of density
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

voxel_object vo;


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




// Replace the do_blackening function with this corrected version
// Color interpolation helper for blackening effect
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
		updateTriangleBuffer(v);
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
								if (v.voxel_blacken_times[voxelIndex] < 0.0f) {
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


#endif