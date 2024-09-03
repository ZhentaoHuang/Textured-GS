/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */
#include <cuda_runtime.h>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;



__device__ float3 getRayVec(float2 pix, int W, int H, const float* viewMatrix, const float* invProj, glm::vec3 campos)
{

	// Convert pixel coordinates to normalized device coordinates (NDC)
	float ndcX = 2.0f * ((pix.x + 1.0f) / W) - 1.0f;
	float ndcY = 2.0f * ((pix.y + 1.0f) / H) - 1.0f;

	// Define point in clip coordinates (z value chosen for a point in front of the camera)
	float p_hom_x_r = ndcX * 1.0000001f;
	float p_hom_y_r = ndcY * 1.0000001f;
	float p_hom_z_r = (100.0f + 0.01f - 1.0f) / (100.0f - 0.01f);
	float4 clipCoords = make_float4(p_hom_x_r, p_hom_y_r, p_hom_z_r, 1.0f);

	// // Transform to camera space using the inverse projection matrix
	// float4 camCoords = transformFloat4_4x4(clipCoords, invProj);
	// float invW = 1.0f / camCoords.w;  // Compute inverse of w once
	// camCoords = make_float4(camCoords.x * invW, camCoords.y * invW, camCoords.z * invW, 1.0f);

	// // Compute the direction vector from the camera position to the point in camera space
	// float3 realVector = make_float3(camCoords.x - campos.x, camCoords.y - campos.y, camCoords.z - campos.z);

	// // Normalize the direction vector
	// float invNorm = 1.0f / sqrt(realVector.x * realVector.x + realVector.y * realVector.y + realVector.z * realVector.z);
	// float3 rayDirection = make_float3(realVector.x * invNorm, realVector.y * invNorm, realVector.z * invNorm);

	// Transform to camera space using the inverse projection matrix
	float4 camCoords = transformFloat4_4x4(clipCoords, invProj);
	camCoords = make_float4(camCoords.x / camCoords.w, camCoords.y / camCoords.w, camCoords.z / camCoords.w, 1.0f);

	// Compute the direction vector from the camera position to the point in camera space
	float3 realVector = make_float3(camCoords.x - campos.x, camCoords.y - campos.y, camCoords.z - campos.z);

	// Normalize the direction vector
	float norm = sqrt(realVector.x * realVector.x + realVector.y * realVector.y + realVector.z * realVector.z);
	float3 rayDirection = make_float3(realVector.x / norm, realVector.y / norm, realVector.z / norm);


	return rayDirection;
}


__device__ void transpose(float *m) {
    float tmp;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            tmp = m[i * 4 + j];
            m[i * 4 + j] = m[j * 4 + i];
            m[j * 4 + i] = tmp;
        }
    }
}



__device__ float determinant(const float *m) {
    return
        m[0] * (m[5] * (m[10] * m[15] - m[11] * m[14]) -
                m[6] * (m[9] * m[15] - m[11] * m[13]) +
                m[7] * (m[9] * m[14] - m[10] * m[13])) -

        m[4] * (m[1] * (m[10] * m[15] - m[11] * m[14]) -
                m[2] * (m[9] * m[15] - m[11] * m[13]) +
                m[3] * (m[9] * m[14] - m[10] * m[13])) +

        m[8] * (m[1] * (m[6] * m[15] - m[7] * m[14]) -
                m[2] * (m[5] * m[15] - m[7] * m[13]) +
                m[3] * (m[5] * m[14] - m[6] * m[13])) -

        m[12] * (m[1] * (m[6] * m[11] - m[7] * m[10]) -
                 m[2] * (m[5] * m[11] - m[7] * m[9]) +
                 m[3] * (m[5] * m[10] - m[6] * m[9]));
}

__device__ void adjugate(const float *m, float *adj) {
    adj[0] = m[5] * (m[10] * m[15] - m[11] * m[14]) - m[6] * (m[9] * m[15] - m[11] * m[13]) + m[7] * (m[9] * m[14] - m[10] * m[13]);
    adj[4] = -(m[1] * (m[10] * m[15] - m[11] * m[14]) - m[2] * (m[9] * m[15] - m[11] * m[13]) + m[3] * (m[9] * m[14] - m[10] * m[13]));
    adj[8] = m[1] * (m[6] * m[15] - m[7] * m[14]) - m[2] * (m[5] * m[15] - m[7] * m[13]) + m[3] * (m[5] * m[14] - m[6] * m[13]);
    adj[12] = -(m[1] * (m[6] * m[11] - m[7] * m[10]) - m[2] * (m[5] * m[11] - m[7] * m[9]) + m[3] * (m[5] * m[10] - m[6] * m[9]));

    adj[1] = -(m[4] * (m[10] * m[15] - m[11] * m[14]) - m[6] * (m[8] * m[15] - m[11] * m[12]) + m[7] * (m[8] * m[14] - m[10] * m[12]));
    adj[5] = m[0] * (m[10] * m[15] - m[11] * m[14]) - m[2] * (m[8] * m[15] - m[11] * m[12]) + m[3] * (m[8] * m[14] - m[10] * m[12]);
    adj[9] = -(m[0] * (m[6] * m[15] - m[7] * m[14]) - m[2] * (m[4] * m[15] - m[7] * m[12]) + m[3] * (m[4] * m[14] - m[6] * m[12]));
    adj[13] = m[0] * (m[6] * m[11] - m[7] * m[10]) - m[2] * (m[4] * m[11] - m[7] * m[8]) + m[3] * (m[4] * m[10] - m[6] * m[8]);

    adj[2] = m[4] * (m[9] * m[15] - m[11] * m[13]) - m[5] * (m[8] * m[15] - m[11] * m[12]) + m[7] * (m[8] * m[13] - m[9] * m[12]);
    adj[6] = -(m[0] * (m[9] * m[15] - m[11] * m[13]) - m[1] * (m[8] * m[15] - m[11] * m[12]) + m[3] * (m[8] * m[13] - m[9] * m[12]));
    adj[10] = m[0] * (m[5] * m[15] - m[7] * m[13]) - m[1] * (m[4] * m[15] - m[7] * m[12]) + m[3] * (m[4] * m[13] - m[5] * m[12]);
    adj[14] = -(m[0] * (m[5] * m[11] - m[7] * m[9]) - m[1] * (m[4] * m[11] - m[7] * m[8]) + m[3] * (m[4] * m[9] - m[5] * m[8]));

    adj[3] = -(m[4] * (m[9] * m[14] - m[10] * m[13]) - m[5] * (m[8] * m[14] - m[10] * m[12]) + m[6] * (m[8] * m[13] - m[9] * m[12]));
    adj[7] = m[0] * (m[9] * m[14] - m[10] * m[13]) - m[1] * (m[8] * m[14] - m[10] * m[12]) + m[2] * (m[8] * m[13] - m[9] * m[12]);
    adj[11] = -(m[0] * (m[5] * m[14] - m[6] * m[13]) - m[1] * (m[4] * m[14] - m[6] * m[12]) + m[2] * (m[4] * m[13] - m[5] * m[12]));
    adj[15] = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8]);
}




__device__ void invertMatrix(const float *mat, float *invMat) {
    float det = determinant(mat);
    if (fabs(det) < 1e-6) {
        // Matrix is singular or nearly singular, handle this case in your application
		printf("abnormal!!!!!\n\n\n");
        return;
    }

    float adj[16];
    adjugate(mat, adj);

    float invDet = 1.0 / det;
    for (int i = 0; i < 16; ++i) {
        invMat[i] = adj[i] * invDet;
    }
}

__device__ void invertAndTransposeMatrix(const float *mat, float *invMat) {
    invertMatrix(mat, invMat);
    transpose(invMat);  // Transpose the matrix to convert between row-major and column-major
}


// __device__ float3 getRayVec(float2 pix, int W, int H, const float* viewMatrix, const float* projMatrix, glm::vec3 campos)
// {
//     // // Convert pixel coordinates to normalized device coordinates (NDC)
//     // float ndcX = 2.0f * ((pix.x + 1.0f) / W) - 1.0f;
//     // float ndcY = 2.0f * ((pix.y + 1.0f) / H) - 1.0f;


// 	// 	//inverse process of 'Transform point by projecting'
// 	// float p_hom_x_r = ndcX*(1.0000001);
// 	// float p_hom_y_r = ndcY*(1.0000001);
// 	// // self.zfar = 100.0, self.znear = 0.01
// 	// float p_hom_z_r = (100+0.01-1)/(100-0.01);
// 	// float p_hom_w_r = 1;

//     // // Define point in clip coordinates as float4 for handling perspective division
//     // // float4 clipCoords = make_float4(ndcX, ndcY, -0.01f, 1.0f);  // w = 1.0f for perspective division
// 	// float4 clipCoords = make_float4(p_hom_x_r, p_hom_y_r, p_hom_z_r, 1.0f);
// 	// // float real_viewMatrix[16];

//     // // Inverse matrices
//     // float invProj[16];
//     // invertAndTransposeMatrix(projMatrix, invProj);


//     // // Transform to camera space using the inverse projection matrix
// 	// float4 camCoords = transformFloat4_4x4(clipCoords, invProj);
// 	// camCoords = {camCoords.x / camCoords.w, camCoords.y / camCoords.w, camCoords.z / camCoords.w, 1};
// 	// float3 realVector = {camCoords.x - campos.x, camCoords.y - campos.y, camCoords.z - campos.z};
// 	// float norm = sqrt(realVector.x * realVector.x + realVector.y * realVector.y + realVector.z * realVector.z);
// 	// // float4 worldCoords = transformFloat4_4x4(camCoords, invView);

// 	// // Normalize to ensure it's a direction vector
//     // float3 rayDirection = make_float3(realVector.x / norm, realVector.y / norm, realVector.z / norm);
//     // // float norm = sqrt(rayDirection.x * rayDirection.x + rayDirection.y * rayDirection.y + rayDirection.z * rayDirection.z);
//     // // rayDirection = make_float3(rayDirection.x / norm, rayDirection.y / norm, rayDirection.z / norm);
// 	// // Minus campos

//     // return rayDirection;
// 	    // Convert pixel coordinates to normalized device coordinates (NDC)
//     float ndcX = 2.0f * ((pix.x + 1.0f) / W) - 1.0f;
//     float ndcY = 2.0f * ((pix.y + 1.0f) / H) - 1.0f;

//     // Define point in clip coordinates (z value chosen for a point in front of the camera)
//     float p_hom_x_r = ndcX * 1.0000001f;
//     float p_hom_y_r = ndcY * 1.0000001f;
//     float p_hom_z_r = (100.0f + 0.01f - 1.0f) / (100.0f - 0.01f);
//     float4 clipCoords = make_float4(p_hom_x_r, p_hom_y_r, p_hom_z_r, 1.0f);

//     // Inverse projection matrix
//     float invProj[16];
//     invertAndTransposeMatrix(projMatrix, invProj);

//     // Transform to camera space using the inverse projection matrix
//     float4 camCoords = transformFloat4_4x4(clipCoords, invProj);
//     camCoords = make_float4(camCoords.x / camCoords.w, camCoords.y / camCoords.w, camCoords.z / camCoords.w, 1.0f);

//     // Compute the direction vector from the camera position to the point in camera space
//     float3 realVector = make_float3(camCoords.x - campos.x, camCoords.y - campos.y, camCoords.z - campos.z);

//     // Normalize the direction vector
// 	float norm = sqrt(realVector.x * realVector.x + realVector.y * realVector.y + realVector.z * realVector.z);
//     float3 rayDirection = make_float3(realVector.x / norm, realVector.y / norm, realVector.z / norm);

//     return rayDirection;
// }



__device__ glm::vec3 getIntersection3D_1(float3 ray, const glm::vec3 mean, const glm::vec4 rot, glm::vec3 campos, glm::vec3 scale) {

	glm::vec3 intersection;

	float3 o_t = {campos.x - mean.x, campos.y - mean.y, campos.z - mean.z};
	float r = rot.x;
    float x = rot.y;
    float y = rot.z;
    float z = rot.w;
    float R[9] = {
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    };

	// Assuming R transforms from local to world, use transpose for world to local
    // Transpose manually for clarity in this example (not done here, assuming R is already set for world to local)
    float3 oLocal = transformR_T(o_t, R); // new center in local space
    float3 dLocal = transformR_T(ray, R); // new direction in local space
	// glm::vec3 scale = s + 0.0000001f;

	float A = 	(dLocal.x * dLocal.x) / (scale.x   + 0.0000001f) + 
				(dLocal.y * dLocal.y) / (scale.y + 0.0000001f) + 
				(dLocal.z * dLocal.z) / (scale.z + 0.0000001f);
	float B = 2 * ((oLocal.x * dLocal.x) / (scale.x + 0.0000001f) + 
					(oLocal.y * dLocal.y) / (scale.y + 0.0000001f) + 
					(oLocal.z * dLocal.z) / (scale.z + 0.0000001f));
	float C = 	(oLocal.x * oLocal.x) / (scale.x + 0.0000001f) + 
				(oLocal.y * oLocal.y) / (scale.y + 0.0000001f) + 
				(oLocal.z * oLocal.z) / (scale.z + 0.0000001f) - 1;

	float discriminant = B * B - 4.f * A * C;

	if (discriminant < 0)
	{
		intersection = glm::vec3(1.0f,1.0f,1.0f);
	}
	else
	{
		float sqrt_discriminant = sqrtf(discriminant);
		float t0 = (-B - sqrt_discriminant) / (2 * A);
		float t1 = (-B + sqrt_discriminant) / (2 * A);

		float t = fminf(t0,t1);
		// if (t0 < t1)
		// {
		// 	t = t0;
		// }
		// else
		// {
		// 	t = t1;
		// }

		intersection = glm::vec3(oLocal.x + t * dLocal.x,
                                      oLocal.y + t * dLocal.y,
                                      oLocal.z + t * dLocal.z);

		intersection = intersection * (1.0f/(sqrt(scale) + 0.0000001f));
		if (glm::length(intersection) != 0)
				intersection = intersection / glm::length(intersection);
		// unit_int.x = intersection.x* (1/(sqrt(collected_scale[j].x) + 0.0000001f));
		// unit_int.y = intersection.y* (1/(sqrt(collected_scale[j].y) + 0.0000001f));
		// unit_int.z = intersection.z* (1/(sqrt(collected_scale[j].z) + 0.0000001f));
	}

	return intersection;
}





__device__ float computeOpacityFromIntersection(int idx, const glm::vec3 unit_int, const float* texture_opacity, const int deg)
{


	float x = unit_int.x;
	float y = unit_int.y;
	float z = unit_int.z;

	int max_coeffs = 16;

	// float sh[16];
	float* sh = ((float*)texture_opacity) + idx * 16;
	// for (int i = 0; i < (deg+1) * (deg+1); i++)
	// {
	// 	// sh[i].x = texture[idx*max_coeffs + 3*i];
	// 	// sh[i].y = texture[idx*max_coeffs + 3*i + 1]; 
	// 	// sh[i].z = texture[idx*max_coeffs + 3*i + 2];
	// 	sh[i] = texture_opacity[idx * max_coeffs + i];
	// }



// 	// l = 0, m = 0
// sh[0] = 0.7f;  // Ambient term, slightly higher to ensure visibility

// // l = 1, m = -1 to 1
// sh[1] = 2.0f;  // Enhance directional lighting strength
// sh[2] = -2.0f; // Strong opposite shadows
// sh[3] = 1.0f;  // Light from another direction

// // l = 2, m = -2 to 2
// sh[4] = 1.0f;
// sh[5] = -1.0f;
// sh[6] = 0.5f;  // Mild vertical illumination
// sh[7] = -0.5f;
// sh[8] = 1.0f;  // Strong horizontal contrast

// // l = 3, m = -3 to 3
// sh[9] = 0.75f;
// sh[10] = -0.75f;
// sh[11] = 0.5f;
// sh[12] = -0.5f;
// sh[13] = 0.5f;
// sh[14] = -0.5f;
// sh[15] = 0.3f; // Subtle high-degree effects


	float result = SH_C0 * sh[0];
	if(deg > 0)
	{
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
	}

	if (deg > 1)
	{
		float xx = x * x, yy = y * y, zz = z * z;
		float xy = x * y, yz = y * z, xz = x * z;
		result = result +
			SH_C2[0] * xy * sh[4] +
			SH_C2[1] * yz * sh[5] +
			SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
			SH_C2[3] * xz * sh[7] +
			SH_C2[4] * (xx - yy) * sh[8];
		
		if (deg > 2)
		{
			result = result +
				SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
				SH_C3[1] * xy * z * sh[10] +
				SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
				SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
				SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
				SH_C3[5] * z * (xx - yy) * sh[14] +
				SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];

		}
	}

	result = 1.0f / (1.0f + glm::exp(-result));

	return result;
}




__device__ glm::vec3 computeColorFromD(int idx, const glm::vec3 unit_int, const float* texture, const int deg)
{

	float x = unit_int.x;
	float y = unit_int.y;
	float z = unit_int.z;

	int max_coeffs = 48;

	
	glm::vec3* sh = ((glm::vec3*)texture) + idx * 16;
	// glm::vec3 sh[16];

	// for (int i = 0; i < (deg+1) * (deg+1); i++)
	// {
	// 	sh[i].x = texture[idx*max_coeffs + 3*i];
	// 	sh[i].y = texture[idx*max_coeffs + 3*i + 1]; 
	// 	sh[i].z = texture[idx*max_coeffs + 3*i + 2];
	// 	// sh[i].x = 0;
	// 	// sh[i].y = 0; 
	// 	// sh[i].z = 0;
	// }



// sh[0].x = 0.42;
// sh[0].y = -0.23;
// sh[0].z = 0.88;

// sh[1].x = -0.95;
// sh[1].y = -0.76;
// sh[1].z = 0.65;

// sh[2].x = 0.34;
// sh[2].y = -0.11;
// sh[2].z = -0.45;

// sh[3].x = 0.29;
// sh[3].y = 0.68;
// sh[3].z = -0.74;

// sh[4].x = -0.12;
// sh[4].y = 0.43;
// sh[4].z = 0.89;

// sh[5].x = -0.26;
// sh[5].y = 0.17;
// sh[5].z = -0.92;

// sh[6].x = 0.57;
// sh[6].y = -0.82;
// sh[6].z = 0.05;

// sh[7].x = 0.76;
// sh[7].y = 0.65;
// sh[7].z = -0.33;

// sh[8].x = -0.49;
// sh[8].y = -0.87;
// sh[8].z = 0.03;

// sh[9].x = 0.95;
// sh[9].y = -0.29;
// sh[9].z = -0.15;

// sh[10].x = -0.75;
// sh[10].y = 0.66;
// sh[10].z = 0.04;

// sh[11].x = -0.98;
// sh[11].y = 0.19;
// sh[11].z = 0.06;

// sh[12].x = -0.31;
// sh[12].y = 0.95;
// sh[12].z = 0.23;

// sh[13].x = 0.47;
// sh[13].y = 0.88;
// sh[13].z = -0.06;

// sh[14].x = 0.15;
// sh[14].y = -0.58;
// sh[14].z = -0.81;

// sh[15].x = -0.21;
// sh[15].y = -0.74;
// sh[15].z = 0.63;





	glm::vec3 result = SH_C0 * sh[0];
	if(deg > 0)
	{
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
	}

	if (deg > 1)
	{
		float xx = x * x, yy = y * y, zz = z * z;
		float xy = x * y, yz = y * z, xz = x * z;
		result = result +
			SH_C2[0] * xy * sh[4] +
			SH_C2[1] * yz * sh[5] +
			SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
			SH_C2[3] * xz * sh[7] +
			SH_C2[4] * (xx - yy) * sh[8];
		
		if (deg > 2)
		{
			result = result +
				SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
				SH_C3[1] * xy * z * sh[10] +
				SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
				SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
				SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
				SH_C3[5] * z * (xx - yy) * sh[14] +
				SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];

		}
	}
	// result += 0.5f;


	

	result = 1.0f / (1.0f + glm::exp(-result));

	return result;


}




// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	// const float* opacities,
	const float* shs,
	// bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// 	printf("Projection Matrix: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f], View Matrix: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n",
    //    projmatrix[0], projmatrix[1], projmatrix[2], projmatrix[3],
    //    projmatrix[4], projmatrix[5], projmatrix[6], projmatrix[7],
    //    projmatrix[8], projmatrix[9], projmatrix[10], projmatrix[11],
    //    projmatrix[12], projmatrix[13], projmatrix[14], projmatrix[15],
    //    viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
    //    viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
    //    viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
    //    viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]);


	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// if (colors_precomp == nullptr)
	// {
	// 	glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
	// 	// result.x = 0;
	// 	// result.y = 0;
	// 	// result.z = 0;
	// 	rgb[idx * C + 0] = result.x;
	// 	rgb[idx * C + 1] = result.y;
	// 	rgb[idx * C + 2] = result.z;
	// }

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, 0.99f};//opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ texture,
	const float* __restrict__ texture_opacity,
	float* rgb,
	// float* __restrict__ pixel_count,
	// float* __restrict__ sig_out,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	// bool* clamped,
	const int degree,
	const float* viewmatrix,
	const float* projmatrix,
	const float* projmatrix_inv,
	const __restrict__  glm::vec3* means3D,
	const __restrict__  glm::vec3* cam_pos,
	const __restrict__  glm::vec4* rotations,
	const __restrict__  glm::vec3* scales
	)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ glm::vec3 collected_scale[BLOCK_SIZE];
	__shared__ glm::vec4 collected_rotation[BLOCK_SIZE];
	__shared__ glm::vec3 collected_mean[BLOCK_SIZE];
	// __shared__ glm::vec3 collected_texture[BLOCK_SIZE *16];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };



	float3 ray = getRayVec(pixf, W, H, viewmatrix, projmatrix_inv, *cam_pos);
	
	// float invProj[16];
    // invertAndTransposeMatrix(projmatrix, invProj);
	// printf("invProj: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\nprojmatrix_inv: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n",
    //    invProj[0], invProj[1], invProj[2], invProj[3],
    //    invProj[4], invProj[5], invProj[6], invProj[7],
    //    invProj[8], invProj[9], invProj[10], invProj[11],
    //    invProj[12], invProj[13], invProj[14], invProj[15],
    //    projmatrix_inv[0], projmatrix_inv[1], projmatrix_inv[2], projmatrix_inv[3],
    //    projmatrix_inv[4], projmatrix_inv[5], projmatrix_inv[6], projmatrix_inv[7],
    //    projmatrix_inv[8], projmatrix_inv[9], projmatrix_inv[10], projmatrix_inv[11],
    //    projmatrix_inv[12], projmatrix_inv[13], projmatrix_inv[14], projmatrix_inv[15]);


	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_scale[block.thread_rank()] = scales[coll_id]; 
			collected_rotation[block.thread_rank()] = rotations[coll_id]; 
			collected_mean[block.thread_rank()] = means3D[coll_id];
			// for (int k = 0; k < 3; k++)
			// {
			// collected_mean[block.thread_rank()].x = means3D[coll_id * 3 + 0];
			// collected_mean[block.thread_rank()].y = means3D[coll_id * 3 + 1];
			// collected_mean[block.thread_rank()].z = means3D[coll_id * 3 + 2];
			// }
			// for (int k = 0; k < 16; k++)
			// {
			// 	collected_texture[k * BLOCK_SIZE + block.thread_rank()].x = texture[coll_id * 48 + 3*k +0];
			// 	collected_texture[k * BLOCK_SIZE + block.thread_rank()].y = texture[coll_id * 48 + 3*k +1];
			// 	collected_texture[k * BLOCK_SIZE + block.thread_rank()].z = texture[coll_id * 48 + 3*k +2];
			// }
			
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];


			// con_o.w = 0.99;



			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			// power = -0.001f;
			// printf("con_o.x: %f, con_o.y: %f, con_o.y: %f", con_o.x, con_o.y, con_o.z);
			if (power > 0.0f)
				continue;


			// glm::vec3 mean = { means3D[3 * collected_id[j]], means3D[3 * collected_id[j] + 1], means3D[3 * collected_id[j] + 2] };
			glm::vec3 mean = collected_mean[j];
			// mean = {0.0f, 0.1f, 0.2f};
			// glm::vec4 rotation = {0.5f, 0.5f, 0.5f, 0.5f};


			// float3 ray = {0.5f,0.5f,0.5f};
			// float3 intersection1 = getIntersection3D_1(ray, mean, rotations[collected_id[j]], *cam_pos, scales[collected_id[j]]);
			glm::vec3 unit_int = getIntersection3D_1(ray, mean, collected_rotation[j], *cam_pos, collected_scale[j]);
			if(unit_int == glm::vec3(1.0f, 1.0f, 1.0f))
				continue;



			con_o.w = computeOpacityFromIntersection(collected_id[j], unit_int, texture_opacity, degree);
			// con_o.w = 0.3;
			// printf("conw: %f\n", con_o.w);
			// if( length_squared == 0)
			// 	con_o.w = 0;
			// Eq. (2) from 3D Gaussian splatting paper.s
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			// if(intersection1.x !=0 || intersection.x !=0)
			// 	printf("int: %f,%f,%f, int1: %f,%f%f, alpha: %f, con_o.w:%f, G%f, s:%f,%f,%f,\n", intersection.x, intersection.y, intersection.z, 
			// 	intersection1.x, intersection1.y, intersection1.z, alpha, con_o.w, exp(power), scales[collected_id[j]].x, scales[collected_id[j]].y,
			// 	scales[collected_id[j]].z);

			// alpha = 0.99f;
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			// glm::vec3 sh[16];
			// glm::vec3* sh = ((glm::vec3*)collected_texture) + j * 16;
			// for (int count = 0; count < 16; count ++)
			// {
			// 	sh[count].x = collected_texture[(3* count + 0)* BLOCK_SIZE + j];
			// 	sh[count].y = collected_texture[(3* count + 1)* BLOCK_SIZE + j];
			// 	sh[count].z = collected_texture[(3* count + 2)* BLOCK_SIZE + j];

			// }
			
			glm::vec3 rgb_out = computeColorFromD(collected_id[j], unit_int, texture, degree);

			// Eq. (3) from 3D Gaussian splatting paper.
			// for (int ch = 0; ch < CHANNELS; ch++)
				// C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;


			// rgb[3 * collected_id[j]]= rgb_out.x;
			// rgb[3 * collected_id[j] + 1] = rgb_out.y;
			// rgb[3 * collected_id[j] + 2] = rgb_out.z;
				
			C[0] += rgb_out.x * alpha * T;
			C[1] += rgb_out.y * alpha * T;
			C[2] += rgb_out.z * alpha * T;


			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float* texture,
	const float* texture_opacity,
	float* rgb,
	// float* pixel_count,
	// float* sig_out,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	// bool* clamped,
	const int D,
	const float* viewmatrix,
	const float* projmatrix,
	const float* projmatrix_inv,
	const glm::vec3* means3D,
	const glm::vec3* cam_pos,
	const glm::vec4* rotations,
	const glm::vec3* scales)
{
	// for (int i = 0; i < 12; i++)
	// {
	// 	printf("%f,", texture[i]);
	// }
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		texture,
		texture_opacity,
		rgb,
		// pixel_count,
		// sig_out,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		// clamped,
		D,
		viewmatrix,
		projmatrix,
		projmatrix_inv,
		means3D,
		cam_pos,
		rotations,
		scales);
			cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}

}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	// const float* opacities,
	const float* shs,
	// bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		// opacities,
		shs,
		// clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}