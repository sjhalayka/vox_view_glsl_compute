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
		int cell_x = static_cast<int>((point.x - vo_grid_min.x) / cell_size);
		int cell_y = static_cast<int>((point.y - vo_grid_min.y) / cell_size);
		int cell_z = static_cast<int>((point.z - vo_grid_min.z) / cell_size);

		// Check bounds
		if (cell_x < 0 || cell_x >= voxel_x_res ||
			cell_y < 0 || cell_y >= voxel_y_res ||
			cell_z < 0 || cell_z >= voxel_z_res) {
			return false;  // Outside grid
		}

		// Find the index in the flattened 3D array
		size_t cell_index = cell_x + (cell_y * voxel_x_res) + (cell_z * voxel_x_res * voxel_y_res);

		long long signed int voxel_idx = vo_grid_cells[cell_index];

		if (voxel_idx == -1)
			return false;  // No voxel here

		// Do a precise check against the voxel
		const float half_size = cell_size * 0.5f;
		const custom_math::vertex_3& center = voxel_centres[voxel_idx];

		if (point.x >= center.x - half_size &&
			point.x <= center.x + half_size &&
			point.y >= center.y - half_size &&
			point.y <= center.y + half_size &&
			point.z >= center.z - half_size &&
			point.z <= center.z + half_size)
		{
			voxel_index = voxel_idx;
			return true;
		}

		return false;
	}


	// Combine the grid with the model transformation
	bool is_point_in_voxel_grid(const custom_math::vertex_3& test_point,
		const glm::mat4& model,
		size_t& voxel_index,
		voxel_object& v) {
		// 1. Calculate the inverse model matrix
		glm::mat4 inv_model_matrix = glm::inverse(model);

		// 2. Transform the test point with the inverse model matrix
		glm::vec4 model_space_point(test_point.x, test_point.y, test_point.z, 1.0f);
		glm::vec4 local_space_point = inv_model_matrix * model_space_point;

		// 3. Create a vertex_3 from the transformed point
		custom_math::vertex_3 transformed_point(
			local_space_point.x,
			local_space_point.y,
			local_space_point.z
		);

		// 4. Use the grid to find the voxel
		return find_voxel_containing_point(transformed_point, voxel_index);
	}




};










voxel_object vo;




//
//void calc_AABB_min_max_locations(void)
//{
//	float x_min = numeric_limits<float>::max();
//	float y_min = numeric_limits<float>::max();
//	float z_min = numeric_limits<float>::max();
//	float x_max = -numeric_limits<float>::max();
//	float y_max = -numeric_limits<float>::max();
//	float z_max = -numeric_limits<float>::max();
//
//	for (size_t t = 0; t < tri_vec.size(); t++)
//	{
//		for (size_t j = 0; j < 3; j++)
//		{
//			if (tri_vec[t].vertex[j].x < x_min)
//				x_min = tri_vec[t].vertex[j].x;
//
//			if (tri_vec[t].vertex[j].x > x_max)
//				x_max = tri_vec[t].vertex[j].x;
//
//			if (tri_vec[t].vertex[j].y < y_min)
//				y_min = tri_vec[t].vertex[j].y;
//
//			if (tri_vec[t].vertex[j].y > y_max)
//				y_max = tri_vec[t].vertex[j].y;
//
//			if (tri_vec[t].vertex[j].z < z_min)
//				z_min = tri_vec[t].vertex[j].z;
//
//			if (tri_vec[t].vertex[j].z > z_max)
//				z_max = tri_vec[t].vertex[j].z;
//		}
//	}
//
//	min_location.x = x_min;
//	min_location.y = y_min;
//	min_location.z = z_min;
//
//	max_location.x = x_max;
//	max_location.y = y_max;
//	max_location.z = z_max;
//}




void centre_voxels_on_xyz(voxel_object &v)
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


bool write_triangles_to_binary_stereo_lithography_file(const vector<custom_math::triangle>& triangles, const char* const file_name)
{
	cout << "Triangle count: " << triangles.size() << endl;

	if (0 == triangles.size())
		return false;

	// Write to file.
	ofstream out(file_name, ios_base::binary);

	if (out.fail())
		return false;

	const size_t header_size = 80;
	vector<char> buffer(header_size, 0);
	const unsigned int num_triangles = static_cast<unsigned int>(triangles.size()); // Must be 4-byte unsigned int.
	custom_math::vertex_3 normal;

	// Write blank header.
	out.write(reinterpret_cast<const char*>(&(buffer[0])), header_size);

	// Write number of triangles.
	out.write(reinterpret_cast<const char*>(&num_triangles), sizeof(unsigned int));

	// Copy everything to a single buffer.
	cout << "Generating normal/vertex/attribute buffer" << endl;

	// Enough bytes for twelve 4-byte floats plus one 2-byte integer, per triangle.
	const size_t data_size = (12 * sizeof(float) + sizeof(short unsigned int)) * num_triangles;
	buffer.resize(data_size, 0);

	// Use a pointer to assist with the copying.
	char* cp = &buffer[0];

	for (vector<custom_math::triangle>::const_iterator i = triangles.begin(); i != triangles.end(); i++)
	{
		// Get face normal.
		custom_math::vertex_3 v0 = i->vertex[1] - i->vertex[0];
		custom_math::vertex_3 v1 = i->vertex[2] - i->vertex[0];
		normal = v0.cross(v1);
		normal.normalize();

		memcpy(cp, &normal.x, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &normal.y, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &normal.z, sizeof(float)); cp += sizeof(float);

		memcpy(cp, &i->vertex[0].x, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[0].y, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[0].z, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[1].x, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[1].y, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[1].z, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[2].x, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[2].y, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[2].z, sizeof(float)); cp += sizeof(float);

		cp += sizeof(short unsigned int);
	}

	cout << "Writing " << data_size / 1048576.0f << " MB of data to binary Stereo Lithography file: " << file_name << endl;

	out.write(reinterpret_cast<const char*>(&buffer[0]), data_size);
	out.close();

	return true;
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




//void index_to_xyz(const size_t index, const size_t x_res, const size_t y_res, size_t& x, size_t& y, size_t& z) 
//{
//	z = index / (x_res * y_res);
//	
//	const size_t remainder = index % (x_res * y_res);
//
//	y = remainder / x_res;
//	x = remainder % x_res;
//}


void get_background_points(voxel_object& v)
{
	float x_grid_min = -x_grid_max;
	float y_grid_min = -y_grid_max;
	float z_grid_min = -z_grid_max;

	v.background_indices.resize(x_res * y_res * z_res);
	v.background_centres.resize(x_res * y_res * z_res);
	v.background_densities.resize(x_res * y_res * z_res);
	v.background_collisions.resize(x_res * y_res * z_res);

	const float x_step_size = (x_grid_max - x_grid_min) / (x_res - 1);
	const float y_step_size = (y_grid_max - y_grid_min) / (y_res - 1);
	const float z_step_size = (z_grid_max - z_grid_min) / (z_res - 1);

	custom_math::vertex_3 Z(x_grid_min, y_grid_min, x_grid_min);

	for (size_t z = 0; z < z_res; z++, Z.z += z_step_size)
	{
		Z.x = x_grid_min;

		for (size_t x = 0; x < x_res; x++, Z.x += x_step_size)
		{
			Z.y = y_grid_min;

			for (size_t y = 0; y < y_res; y++, Z.y += y_step_size)
			{
				const custom_math::vertex_3 test_point(Z.x, Z.y, Z.z);

				const size_t index = x + (y * x_res) + (z * x_res * y_res);

				size_t voxel_index = 0;

				v.background_centres[index] = test_point;
				v.background_indices[index] = glm::ivec3(x, y, z);

				if (v.is_point_in_voxel_grid(test_point, v.model_matrix, voxel_index, v))
				{
					v.background_densities[index] = 1.0;
					v.background_collisions[index] = voxel_index;
				}
				else
				{
					v.background_densities[index] = 0.0;
				}
			}
		}
	}

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
	v.background_surface_indices.clear();
	v.background_surface_indices.resize(x_res * y_res * z_res);
	v.background_surface_centres.clear();
	v.background_surface_centres.resize(x_res * y_res * z_res);
	v.background_surface_densities.clear();
	v.background_surface_densities.resize(x_res * y_res * z_res);
	v.background_surface_collisions.clear();
	v.background_surface_collisions.resize(x_res * y_res * z_res);

	// Check each point in the background grid
	for (size_t i = 0; i < v.background_centres.size(); i++)
	{
		// Skip points that are already inside the voxel grid
		if (v.background_densities[i] > 0)
			continue;

		// Get the grid coordinates for this point
		const int x = v.background_indices[i].x;
		const int y = v.background_indices[i].y;
		const int z = v.background_indices[i].z;

		const size_t index = x + (y * x_res) + (z * x_res * y_res);

		// Check all 6 adjacent neighbors

		bool is_surface = false;

		for (int dir = 0; dir < 6; dir++)
		{
			const int nx = x + directions[dir][0];
			const int ny = y + directions[dir][1];
			const int nz = z + directions[dir][2];

			// Skip if neighbor is outside the grid
			if (nx < 0 || nx >= static_cast<int>(x_res) ||
				ny < 0 || ny >= static_cast<int>(y_res) ||
				nz < 0 || nz >= static_cast<int>(z_res))
			{
				continue;
			}

			// Calculate the index of the neighboring point
			size_t neighbor_index = nx + (ny * x_res) + (nz * x_res * y_res);

			// If the neighboring point is inside the voxel grid, this is a surface point
			if (neighbor_index < v.background_densities.size() && v.background_densities[neighbor_index] > 0)
			{
				is_surface = true;

				const size_t collision = v.background_collisions[neighbor_index];

				v.background_surface_collisions[index].push_back(collision);
			}
		}


		v.background_surface_indices[index] = v.background_indices[i];
		v.background_surface_centres[index] = v.background_centres[i];

		if (is_surface)
		{
			//cout << background_surface_collisions[index].size() << endl;
			v.background_surface_densities[index] = 1.0;
		}
		else
		{
			v.background_surface_densities[index] = 0.0;
		}
	}


}



void get_surface_points(void)
{


//	std::cout << "Found " << background_surface_centres.size() << " surface points" << std::endl;
}




void do_blackening(voxel_object &v)
{
	for (size_t x = 0; x < x_res; x++)
	{
		for (size_t y = 0; y < y_res; y++)
		{
			for (size_t z = 0; z < z_res; z++)
			{
				const size_t index = x + y * x_res + z * x_res * y_res;

				if (v.background_surface_densities[index] == 0.0)
					continue;

				for (size_t i = 0; i < v.background_surface_collisions[index].size(); i++)
				{
					v.voxel_colours[v.background_surface_collisions[index][i]].r *= test_texture[index] / 255.0f;
					v.voxel_colours[v.background_surface_collisions[index][i]].g *= test_texture[index] / 255.0f;
					v.voxel_colours[v.background_surface_collisions[index][i]].b *= test_texture[index] / 255.0f;
					v.voxel_colours[v.background_surface_collisions[index][i]].a = 1.0f;
				}

				//cout << background_surface_collisions[index].size() << endl;

				//if (y >= voxel_y_res / 2)
				//	test_texture[voxel_index] = 1.0;
				//else
				//	test_texture[voxel_index] = 0.0;
			}
		}
	}


}



#endif
