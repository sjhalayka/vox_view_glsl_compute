// Shawn Halayka
// This code and data is in the public domain.


#include "uv_camera.h"




uv_camera::uv_camera(void)
{
	u = v = 0;
	w = 2;
	fov = 45.0f;
	near_plane = 0.01f;
	far_plane = 100.0f;
	win_x = win_y = 0;

	transform();
}

void uv_camera::calculate_camera_matrices(const int width_px, const int height_px)
{
	static const float lock = uv_epsilon * 1000.0f;

	if (u < -uv_pi_half + lock)
		u = -uv_pi_half + lock;
	else if (u > uv_pi_half - lock)
		u = uv_pi_half - lock;

	while (v < 0)
		v += uv_pi_2;

	while (v > uv_pi_2)
		v -= uv_pi_2;

	if (w < 0)
		w = 0;
	else if (w > 10000)
		w = 10000;

	win_x = width_px;
	win_y = height_px;

	transform();
}

void uv_camera::transform(void)
{
	reset();
	rotate();
	translate();

	if (win_y > 0)
	{
		//projection_mat = perspective(
		//	glm::radians(fov),
		//	float(win_x) / float(win_y),
		//	near_plane,
		//	far_plane
		//);

		projection_mat = glm::ortho(0.0f, (float)win_x / 10.0f,
			0.0f, (float)win_y / 10.0f,
			0.0001f, 10000.0f);
	}

	view_mat = glm::lookAt(
		vec3(0, 0, 200),
		vec3(0, 0, 0),
		vec3(0, 1, 0)
	);
}

void uv_camera::reset(void)
{
	eye = vec3(0, 0, 0);
	look_at = vec3(0, 0, 0);
	up = vec3(0, 0, 0);

	look_at.z = -1;
	up.y = 1;
}


vec3 rotate_x(vec3 v, const float& radians)
{
	float t_y = v.y;

	v.y = t_y * cos(radians) + v.z * sin(radians);
	v.z = t_y * -sin(radians) + v.z * cos(radians);

	return v;
}

vec3  rotate_y(vec3 v, const float& radians)
{
	float t_x = v.x;

	v.x = t_x * cos(radians) + v.z * -sin(radians);
	v.z = t_x * sin(radians) + v.z * cos(radians);

	return v;
}

void uv_camera::rotate(void)
{
	// Rotate about the world x axis
	look_at = rotate_x(look_at, u);
	up = rotate_x(up, u);

	// Rotate about the world y axis
	look_at = rotate_y(look_at, v);
	up = rotate_y(up, v);
}


void uv_camera::translate(void)
{
	// Place the eye directly across the sphere from the look-at vector's "tip",
	// Then scale the sphere radius by w
	eye.x = -look_at.x * w;
	eye.y = -look_at.y * w;
	eye.z = -look_at.z * w;
}
