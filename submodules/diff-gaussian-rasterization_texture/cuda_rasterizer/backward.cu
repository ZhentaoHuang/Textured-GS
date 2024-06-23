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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


__device__ void transpose_b(float *m) {
    float tmp;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            tmp = m[i * 4 + j];
            m[i * 4 + j] = m[j * 4 + i];
            m[j * 4 + i] = tmp;
        }
    }
}



__device__ float determinant_b(const float *m) {
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

__device__ void adjugate_b(const float *m, float *adj) {
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




__device__ void invertMatrix_b(const float *mat, float *invMat) {
    float det = determinant_b(mat);
    if (fabs(det) < 1e-6) {
        // Matrix is singular or nearly singular, handle this case in your application
		printf("abnormal!!!!!\n\n\n");
        return;
    }

    float adj[16];
    adjugate_b(mat, adj);

    float invDet = 1.0 / det;
    for (int i = 0; i < 16; ++i) {
        invMat[i] = adj[i] * invDet;
    }
}

__device__ void invertAndTransposeMatrix_b(const float *mat, float *invMat) {
    invertMatrix_b(mat, invMat);
    transpose_b(invMat);  // Transpose the matrix to convert between row-major and column-major
}


__device__ float3 getRayVec_b(float2 pix, int W, int H, const float* viewMatrix, const float* projMatrix, glm::vec3 campos)
{
    // Convert pixel coordinates to normalized device coordinates (NDC)
    float ndcX = 2.0f * ((pix.x + 1.0f) / W) - 1.0f;
    float ndcY = 2.0f * ((pix.y + 1.0f) / H) - 1.0f;


		//inverse process of 'Transform point by projecting'
	float p_hom_x_r = ndcX*(1.0000001);
	float p_hom_y_r = ndcY*(1.0000001);
	// self.zfar = 100.0, self.znear = 0.01
	float p_hom_z_r = (100+0.01-1)/(100-0.01);
	float p_hom_w_r = 1;

    // Define point in clip coordinates as float4 for handling perspective division
    // float4 clipCoords = make_float4(ndcX, ndcY, -0.01f, 1.0f);  // w = 1.0f for perspective division
	float4 clipCoords = make_float4(p_hom_x_r, p_hom_y_r, p_hom_z_r, 1.0f);
	// float real_viewMatrix[16];

	// for (int i = 0; i < 16; i++)
	// {
	// 	real_viewMatrix[i] = viewMatrix[i];
	// }
	// real_viewMatrix[3] = -campos.x;
	// real_viewMatrix[7] = -campos.y;
	// real_viewMatrix[11] = -campos.z;

    // Inverse matrices
    float invProj[16], invView[16];
    invertAndTransposeMatrix_b(projMatrix, invProj);
    invertMatrix_b(viewMatrix, invView);

	// printf("Projection Matrix: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f], View Matrix: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n",
    //    projMatrix[0], projMatrix[1], projMatrix[2], projMatrix[3],
    //    projMatrix[4], projMatrix[5], projMatrix[6], projMatrix[7],
    //    projMatrix[8], projMatrix[9], projMatrix[10], projMatrix[11],
    //    projMatrix[12], projMatrix[13], projMatrix[14], projMatrix[15],
    //    invProj[0], invProj[1], invProj[2], invProj[3],
    //    invProj[4], invProj[5], invProj[6], invProj[7],
    //    invProj[8], invProj[9], invProj[10], invProj[11],
    //    invProj[12], invProj[13], invProj[14], invProj[15]);

	// float4 meanWorld = transformPoint4x4(mean , projMatrix);
	// float3 mean3d = {meanWorld.x/meanWorld.w, meanWorld.y/meanWorld.w, meanWorld.z/meanWorld.w};
	// float2 point_image = { ndc2Pix(mean3d.x, W), ndc2Pix(mean3d.y, H) };

	// float2 ndc = {2.0f * ((point_image.x + 0.5f) / W) - 1.0f, 2.0f * ((point_image.y + 0.5f) / H) - 1.0f};
	// float4 clip = make_float4(ndc.x, ndc.y, -0.01f, 1.0f);  // w = 1.0f for perspective division
	// float4 world = transformPoint4x4(make_float3(clipCoords.x, clipCoords.y, clipCoords.z), invProj);
	// float3 result = make_float3(world.x/world.w, world.y/world.w, world.z/world.w);
	// printf("Point Image: (%.2f, %.2f), Original: (%.2f, %.2f, %.2f), Recovered: (%.2f, %.2f, %.2f)\n",
    //    point_image.x, point_image.y, mean.x, mean.y, mean.z, result.x, result.y, result.z);

	
    // Transform to camera space using the inverse projection matrix
    // float4 worldCoords = transformPoint4x4(make_float3(clipCoords.x, clipCoords.y, clipCoords.z), invProj);
	float4 camCoords = transformFloat4_4x4(clipCoords, invProj);
	// float norm_cam = sqrt(camCoords.x * camCoords.x + camCoords.y * camCoords.y + camCoords.z * camCoords.z);
	camCoords = {camCoords.x / camCoords.w, camCoords.y / camCoords.w, camCoords.z / camCoords.w, 1};
	float3 realVector = {camCoords.x - campos.x, camCoords.y - campos.y, camCoords.z - campos.z};
	float norm = sqrt(realVector.x * realVector.x + realVector.y * realVector.y + realVector.z * realVector.z);
	// float4 worldCoords = transformFloat4_4x4(camCoords, invView);

	// Normalize to ensure it's a direction vector
    float3 rayDirection = make_float3(realVector.x / norm, realVector.y / norm, realVector.z / norm);
    // float norm = sqrt(rayDirection.x * rayDirection.x + rayDirection.y * rayDirection.y + rayDirection.z * rayDirection.z);
    // rayDirection = make_float3(rayDirection.x / norm, rayDirection.y / norm, rayDirection.z / norm);
	// Minus campos

	// if (projMatrix[0] - 0.937958 < 0.00001f && projMatrix[0] - 0.937958 > -0.00001f && pix.x < 348 && pix.x >346 && pix.y < 266 && pix.y > 264)
	// {

	
	// //  printf("Pixel: (%.2f, %.2f), clipCoords: (%.4f, %.4f, %.4f, %.4f),campos: (%.4f, %.4f, %.4f), camCoords: (%.4f, %.4f, %.4f, %.4f), "
    // //        "World: (%.4f, %.4f, %.4f, %.4f), Ray: (%.4f, %.4f, %.4f), proj[0]: %f, rot: (%.4f, %.4f, %.4f, %.4f), mean: (%.4f, %.4f, %.4f)\n",
    // //        pix.x, pix.y, clipCoords.x, clipCoords.y, clipCoords.z, clipCoords.w,campos.x, campos.y, campos.z, camCoords.x, camCoords.y, camCoords.z, camCoords.w,
    // //        worldCoords.x, worldCoords.y, worldCoords.z, worldCoords.w,
    // //        rayDirection.x, rayDirection.y, rayDirection.z, projMatrix[0], rot.x, rot.y, rot.z, rot.w, mean.x, mean.y, mean.z);

	// // 	   	printf("Projection Matrix: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f], inv Matrix: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]\n",
    // //    projMatrix[0], projMatrix[1], projMatrix[2], projMatrix[3],
    // //    projMatrix[4], projMatrix[5], projMatrix[6], projMatrix[7],
    // //    projMatrix[8], projMatrix[9], projMatrix[10], projMatrix[11],
    // //    projMatrix[12], projMatrix[13], projMatrix[14], projMatrix[15],
    // //    invProj[0], invProj[1], invProj[2], invProj[3],
    // //    invProj[4], invProj[5], invProj[6], invProj[7],
    // //    invProj[8], invProj[9], invProj[10], invProj[11],
    // //    invProj[12], invProj[13], invProj[14], invProj[15]);
	// }
    return rayDirection;
}



// __device__ float3 mat3x3_mul_vec3(float R[9], float3 v) {
//     return make_float3(
//         R[0] * v.x + R[3] * v.y + R[6] * v.z,
//         R[1] * v.x + R[4] * v.y + R[7] * v.z,
//         R[2] * v.x + R[5] * v.y + R[8] * v.z
//     );
// }


__device__ float3 getIntersection_b(float3 ray, const float3 mean, const glm::vec4 rot, glm::vec3 campos) {
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
    float zMean = 0;

    if (fabs(dLocal.z) < 1e-6) { // Check if the ray is parallel to the plane
        if (fabs(oLocal.z - zMean) < 1e-6) {
            return oLocal; // Ray lies in the plane
        } else {
            return make_float3(0, 0, 0); // No intersection
        }
    }
    float t = (zMean - oLocal.z) / dLocal.z; // Solve for t
    float3 intersection = make_float3(oLocal.x + t * dLocal.x,
                                      oLocal.y + t * dLocal.y,
                                      oLocal.z + t * dLocal.z);

	// if(oLocal.z < 0)
	// {
	// 	intersection.x = -intersection.x;
	// 	intersection.y = intersection.y;
	// 	intersection.z = intersection.z;
	// }

    return intersection;
}




__device__ float3 getIntersection3D_b(float3 ray, const float3 mean, const glm::vec4 rot, glm::vec3 campos, glm::vec3 scale) {

	float3 intersection;

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
	// float A = 	(dLocal.x * dLocal.x) / (scale.x * scale.x *9.0f  + 0.0000001f) + 
	// 			(dLocal.y * dLocal.y) / (scale.y * scale.y *9.0f+ 0.0000001f) + 
	// 			(dLocal.z * dLocal.z) / (scale.z * scale.z *9.0f+ 0.0000001f);
	// float B = 2 * ((oLocal.x * dLocal.x) / (scale.x * scale.x *9.0f+ 0.0000001f) + 
	// 				(oLocal.y * dLocal.y) / (scale.y * scale.y *9.0f+ 0.0000001f) + 
	// 				(oLocal.z * dLocal.z) / (scale.z * scale.z *9.0f+ 0.0000001f));
	// float C = 	(oLocal.x * oLocal.x) / (scale.x * scale.x *9.0f+ 0.0000001f) + 
	// 			(oLocal.y * oLocal.y) / (scale.y * scale.y *9.0f+ 0.0000001f) + 
	// 			(oLocal.z * oLocal.z) / (scale.z * scale.z *9.0f+ 0.0000001f) - 1;


	float A = 	(dLocal.x * dLocal.x) / (scale.x   + 0.0000001f) + 
				(dLocal.y * dLocal.y) / (scale.y + 0.0000001f) + 
				(dLocal.z * dLocal.z) / (scale.z + 0.0000001f);
	float B = 2 * ((oLocal.x * dLocal.x) / (scale.x + 0.0000001f) + 
					(oLocal.y * dLocal.y) / (scale.y + 0.0000001f) + 
					(oLocal.z * dLocal.z) / (scale.z + 0.0000001f));
	float C = 	(oLocal.x * oLocal.x) / (scale.x + 0.0000001f) + 
				(oLocal.y * oLocal.y) / (scale.y + 0.0000001f) + 
				(oLocal.z * oLocal.z) / (scale.z + 0.0000001f) - 1;

	float discriminant = B * B - 4 * A * C;

	if (discriminant < 0)
	{
		intersection = make_float3(0.0f,0.0f,0.0f);
	}
	else
	{
		double sqrt_discriminant = std::sqrt(discriminant);
		float t0 = (-B - sqrt_discriminant) / (2 * A);
		float t1 = (-B + sqrt_discriminant) / (2 * A);

		float t = 0;
		if (t0 < t1)
		{
			t = t0;
		}
		else
		{
			t = t1;
		}

		intersection = make_float3(oLocal.x + t * dLocal.x,
                                      oLocal.y + t * dLocal.y,
                                      oLocal.z + t * dLocal.z);
	}

	



	return intersection;
}





__device__ float computeOpacityFromIntersection_f(int idx, const glm::vec3 unit_int, const float* texture_opacity, const int deg)
{

	float x = unit_int.x;
	float y = unit_int.y;
	float z = unit_int.z;



	int max_coeffs = 16;

	// const float* sh = texture_opacity;



	float sh[16];

	#pragma unroll
	for (int i = 0; i < (deg+1) * (deg+1); i++)
	{
		sh[i] = texture_opacity[idx*max_coeffs + i];

	}



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





__device__ void computeOpacityFromIntersection(int idx, const glm::vec3 unit_int, float dL_dopacity, float* dL_dtext_opacity, const bool* clamped, glm::vec3* dL_dscales, const float sig_out, const int deg)
{



	int max_coeffs = 16;
	// int deg = 1;
	
	float dL_dRGB = dL_dopacity;

	dL_dRGB *= sig_out * (1 - sig_out);

	float x = unit_int.x;
	float y = unit_int.y;
	float z = unit_int.z;

	

	// float sh[16];

	// #pragma unroll
	// for (int i = 0; i < (deg+1) * (deg+1); i++)
	// {
	// 	// sh[i].x = texture[idx*max_coeffs + 3*i];
	// 	// sh[i].y = texture[idx*max_coeffs + 3*i + 1]; 
	// 	// sh[i].z = texture[idx*max_coeffs + 3*i + 2];
	// 	sh[i] = texture_opacity[i];
	// }

	float dL_dsh[16];


	dL_dsh[0] = SH_C0 * dL_dRGB;

	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;


		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;
			}
		}
	}

	for (int i = 0; i < (deg+1)*(deg+1); i++)
	{

		// dL_dtext[idx*max_coeffs + 3*i] += dL_dsh[i].x;
		// dL_dtext[idx*max_coeffs + 3*i + 1] += dL_dsh[i].y;
		// dL_dtext[idx*max_coeffs + 3*i + 2] += dL_dsh[i].z;

		atomicAdd(&dL_dtext_opacity[idx*max_coeffs + i], dL_dsh[i]);
		// atomicAdd(&dL_dtext[idx*max_coeffs + 3*i + 1], dL_dsh[i].y);
		// atomicAdd(&dL_dtext[idx*max_coeffs + 3*i + 2], dL_dsh[i].z);

	}
}




__device__ glm::vec3 computeColorFromD_f(int idx, const glm::vec3 unit_int, const float* texture, const int deg)
{

	float x = unit_int.x;
	float y = unit_int.y;
	float z = unit_int.z;


	int max_coeffs = 48;

	
	// glm::vec3* sh = ((glm::vec3*)texture) + idx;// * max_coeffs;
	glm::vec3 sh[16];

	#pragma unroll
	for (int i = 0; i < (deg+1) * (deg+1); i++)
	{
		sh[i].x = texture[idx*max_coeffs + 3*i];
		sh[i].y = texture[idx*max_coeffs + 3*i + 1]; 
		sh[i].z = texture[idx*max_coeffs + 3*i + 2];
	}






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


	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.

	// clamped[3 * (idx) + 0] = clamped[3 * (idx) + 0] || (result.x < 0);
	// clamped[3 * (idx) + 1] = clamped[3 * (idx) + 1] || (result.y < 0);
	// clamped[3 * (idx) + 2] = clamped[3 * (idx) + 2] || (result.z < 0);

	result = 1.0f / (1.0f + glm::exp(-result));
	// result += 0.5f;
	// sig_out[3 * idx + 0] += result.x;
	// sig_out[3 * idx + 1] += result.y;
	// sig_out[3 * idx + 2] += result.z;
	// result.x = 1.0f - length_squared;
	// result.y = 1.0f - length_squared;
	// result.z = 1.0f - length_squared;
	// printf("z: %f",z );
	// result.z = z;
	//  d_out[idx] = 1.0 / (1.0 + exp(-result));glm::
	// return glm::max(result, 0.0f);
	return result;


}




__device__ void computeColorFromD(int idx, const glm::vec3 unit_int, const float* texture, glm::vec3 dL_dcolor, float* dL_dtext, const bool* clamped, glm::vec3* dL_dscales, const glm::vec3 sig_out, const int deg)
{



	int max_coeffs = 48;
	// int deg = 1;
	
	glm::vec3 dL_dRGB = dL_dcolor;

	dL_dRGB.x *= sig_out.x * (1 - sig_out.x) ;
	dL_dRGB.y *= sig_out.y * (1 - sig_out.y);
	dL_dRGB.z *= sig_out.z * (1 - sig_out.z);


	float x = unit_int.x;
	float y = unit_int.y;
	float z = unit_int.z;


	// glm::vec3 sh[16];

	// for (int i = 0; i < (deg+1) * (deg+1); i++)
	// {
	// 	sh[i].x = texture[idx*max_coeffs + 3*i];
	// 	sh[i].y = texture[idx*max_coeffs + 3*i + 1]; 
	// 	sh[i].z = texture[idx*max_coeffs + 3*i + 2];
	// }

	glm::vec3 dL_dsh[16];


	dL_dsh[0] = SH_C0 * dL_dRGB;

	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		// dRGBdx = -SH_C1 * sh[3];
		// dRGBdy = -SH_C1 * sh[1];
		// dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			// dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			// dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			// dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				// dRGBdx += (
				// 	SH_C3[0] * sh[9] * 3.f * 2.f * xy +
				// 	SH_C3[1] * sh[10] * yz +
				// 	SH_C3[2] * sh[11] * -2.f * xy +
				// 	SH_C3[3] * sh[12] * -3.f * 2.f * xz +
				// 	SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
				// 	SH_C3[5] * sh[14] * 2.f * xz +
				// 	SH_C3[6] * sh[15] * 3.f * (xx - yy));

				// dRGBdy += (
				// 	SH_C3[0] * sh[9] * 3.f * (xx - yy) +
				// 	SH_C3[1] * sh[10] * xz +
				// 	SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
				// 	SH_C3[3] * sh[12] * -3.f * 2.f * yz +
				// 	SH_C3[4] * sh[13] * -2.f * xy +
				// 	SH_C3[5] * sh[14] * -2.f * yz +
				// 	SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				// dRGBdz += (
				// 	SH_C3[1] * sh[10] * xy +
				// 	SH_C3[2] * sh[11] * 4.f * 2.f * yz +
				// 	SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
				// 	SH_C3[4] * sh[13] * 4.f * 2.f * xz +
				// 	SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	for (int i = 0; i < (deg+1)*(deg+1); i++)
	{

		// dL_dtext[idx*max_coeffs + 3*i] += dL_dsh[i].x;
		// dL_dtext[idx*max_coeffs + 3*i + 1] += dL_dsh[i].y;
		// dL_dtext[idx*max_coeffs + 3*i + 2] += dL_dsh[i].z;

		atomicAdd(&dL_dtext[idx*max_coeffs + 3*i], dL_dsh[i].x);
		atomicAdd(&dL_dtext[idx*max_coeffs + 3*i + 1], dL_dsh[i].y);
		atomicAdd(&dL_dtext[idx*max_coeffs + 3*i + 2], dL_dsh[i].z);

	}

    // Compute forward values of x, y, z as in the original function
    // float x = intersection.x * (1/sqrt(scale.x));
    // float y = intersection.y * (1/sqrt(scale.y));
    // float length_squared = x * x + y * y;
    // float z = (length_squared > 1.0f) ? 0.0f : sqrt(1.0f - length_squared);


	// + 0.3???
    // Compute gradients of x, y, z with respect to scale
    // float dx_dscalex = -0.5 * intersection.x / (scale.x * sqrt(scale.x));
    // float dy_dscaley = -0.5 * intersection.y / (scale.y * sqrt(scale.y));
    // float dz_dscalex = (length_squared > 1.0f) ? 0 : -0.5 / sqrt(1.0f - length_squared) * 2 * x * dx_dscalex;
    // float dz_dscaley = (length_squared > 1.0f) ? 0 : -0.5 / sqrt(1.0f - length_squared) * 2 * y * dy_dscaley;

    // // Compute gradients of the loss with respect to scale
	// glm::vec3 dLdscale;
	// // dL_dx = dL_dRGB * dRGB_dx + 
	// float dL_dx = dL_dRGB.x * dRGBdx.x + dL_dRGB.y * dRGBdx.y + dL_dRGB.z * dRGBdx.z;
	// float dL_dy = dL_dRGB.x * dRGBdy.x + dL_dRGB.y * dRGBdy.y + dL_dRGB.z * dRGBdy.z;
	// float dL_dz = dL_dRGB.x * dRGBdz.x + dL_dRGB.y * dRGBdz.y + dL_dRGB.z * dRGBdz.z;
    // dLdscale.x = dL_dx * dx_dscalex + dL_dz * dz_dscalex;
    // dLdscale.y = dL_dy * dy_dscaley + dL_dz * dz_dscaley;
	// dLdscale.z = 0;


	// glm::vec3 dL_dscale = dL_dscales[idx];
	// dL_dscale.x = dL_dx * dx_dscalex + dL_dz * dz_dscalex;
	// dL_dscale.y = dL_dy * dy_dscaley + dL_dz * dz_dscaley;

	// glm::vec3* dL_dscale = dL_dscales + idx;
	// printf(":%f, %f %f, %f\n", dL_dscale->x, dL_dx, dx_dscalex, dL_dRGB.x );


}





// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x += glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y += glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z += glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	// if (shs)
		// computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ texture,
	const float* __restrict__ texture_opacity,
	const float* __restrict__ sig_out,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dtext,
	float* __restrict__ dL_dtext_opacity,
	const bool* clamped,
	const int D,
	const float* viewmatrix,
	const float* projmatrix,
	const float3* means3D,
	const glm::vec3* cam_pos,
	const glm::vec4* rotations,
	const glm::vec3* scales,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	// __shared__ float collected_texture_opa[16 * BLOCK_SIZE];
	__shared__ float collected_scales[3 * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;


 	glm::mat4 matrix = glm::make_mat4x4(projmatrix);
    glm::mat4 matrix_temp = glm::inverse(matrix);
	float *projmatrix_inv= glm::value_ptr(matrix_temp);

	glm::vec3 ray_origin = *cam_pos;
	glm::vec3 point_rec = {0,0,0};





	float3 p_proj_r = { Pix2ndc(pixf.x, W), Pix2ndc(pixf.y, H), 1};

	//inverse process of 'Transform point by projecting'
	float p_hom_x_r = p_proj_r.x*(1.0000001);
	float p_hom_y_r = p_proj_r.y*(1.0000001);
	// self.zfar = 100.0, self.znear = 0.01
	float p_hom_z_r = (100+0.01-1)/(100-0.01);
	float p_hom_w_r = 1;


	float3 p_hom_r={p_hom_x_r, p_hom_y_r, p_hom_z_r};
	float4 p_orig_r=transformPoint4x4(p_hom_r, projmatrix_inv);
	float p_w = 1.0f / (p_orig_r.w + 0.0000001f);
	float3 p_proj = { p_hom_r.x * p_w, p_hom_r.y * p_w, p_hom_r.z * p_w };

	glm::vec3 ray_direction={
		p_proj.x-ray_origin.x,
		p_proj.y-ray_origin.y,
		p_proj.z-ray_origin.z,
	};

	glm::vec3 normalized_ray_direction = glm::normalize(ray_direction);


	
	float3 ray = getRayVec_b(pixf, W, H, viewmatrix, projmatrix, *cam_pos);

	// printf("ray: %f,%f,%f, ray_n: %f,%f,%f\n", ray.x, ray.y,ray.z, normalized_ray_direction.x, normalized_ray_direction.y, normalized_ray_direction.z);

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			// for (int i = 0; i < C; i++)
			// 	collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			// for (int i = 0; i < 16; i++)
			// 	collected_texture_opa[i * BLOCK_SIZE + block.thread_rank()] = texture_opacity[coll_id * 16 + i]
			// for (int i = 0; i < 3; i++)
			// collected_scales[0 * BLOCK_SIZE + block.thread_rank()] = scales[coll_id].x;
			// collected_scales[1 * BLOCK_SIZE + block.thread_rank()] = scales[coll_id].y;
			// collected_scales[2 * BLOCK_SIZE + block.thread_rank()] = scales[coll_id].z;
			
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;
			// float3 scale;
			// scale.x = collected_scales[j];
			// scale.y = collected_scales[1 * BLOCK_SIZE + j];
			// scale.z = collected_scales[2 * BLOCK_SIZE + j];



			// // compute Gaussian depth
			// // Normalize quaternion to get valid rotation
			// glm::vec4 q = rotations[collected_id[j]];// / glm::length(rot);
			// float rot_r = q.x;
			// float rot_x = q.y;
			// float rot_y = q.z;
			// float rot_z = q.w;


			// // Compute rotation matrix from quaternion
			// glm::mat3 R = glm::mat3(
			// 	1.f - 2.f * (rot_y * rot_y + rot_z * rot_z), 2.f * (rot_x * rot_y - rot_r * rot_z), 2.f * (rot_x * rot_z + rot_r * rot_y),
			// 	2.f * (rot_x * rot_y + rot_r * rot_z), 1.f - 2.f * (rot_x * rot_x + rot_z * rot_z), 2.f * (rot_y * rot_z - rot_r * rot_x),
			// 	2.f * (rot_x * rot_z - rot_r * rot_y), 2.f * (rot_y * rot_z + rot_r * rot_x), 1.f - 2.f * (rot_x * rot_x + rot_y * rot_y)
			// );


			// glm::vec3 temp={
			// 	ray_origin.x-means3D[collected_id[j]].x,
			// 	ray_origin.y-means3D[collected_id[j]].y,
			// 	ray_origin.z-means3D[collected_id[j]].z,
			// };
			// glm::vec3 rotated_ray_origin = R * temp;
			// glm::vec3 rotated_ray_direction = R * normalized_ray_direction;


			// glm::vec3 a_t= rotated_ray_direction/(scales[collected_id[j]]*3.0f)*rotated_ray_direction/(scales[collected_id[j]]*3.0f);
			// float a = a_t.x + a_t.y + a_t.z;

			// glm::vec3 b_t= rotated_ray_direction/(scales[collected_id[j]]*3.0f)*rotated_ray_origin/(scales[collected_id[j]]*3.0f);
			// float b = 2*(b_t.x + b_t.y + b_t.z);

			// glm::vec3 c_t= rotated_ray_origin/(scales[collected_id[j]]*3.0f)*rotated_ray_origin/(scales[collected_id[j]]*3.0f);
			// float c = c_t.x + c_t.y + c_t.z-1;


			// float discriminant=b*b-4*a*c;	
			// float3 intersection;

			// if (discriminant < 0)
			// {
			// 	intersection = make_float3(0.0f,0.0f,0.0f);
			// }
			// else
			// {
			// 	double sqrt_discriminant = std::sqrt(discriminant);
			// 	float t0 = (-b - sqrt_discriminant) / (2 * a);
			// 	float t1 = (-b + sqrt_discriminant) / (2 * a);

			// 	float t = 0;
			// 	if (t0 < t1)
			// 	{
			// 		t = t0;
			// 	}
			// 	else
			// 	{
			// 		t = t1;
			// 	}

			// 	intersection = make_float3(rotated_ray_origin.x + t * rotated_ray_direction.x,
			// 								rotated_ray_origin.y + t * rotated_ray_direction.y,
			// 								rotated_ray_origin.z + t * rotated_ray_direction.z);
			// }


			float3 mean = { means3D[collected_id[j]].x, means3D[collected_id[j]].y, means3D[collected_id[j]].z };
			float3 intersection = getIntersection3D_b(ray, mean, rotations[collected_id[j]], *cam_pos, scales[collected_id[j]]);
			// if (intersection.x != 0)
			// 	printf("int1:%f, %f, %f, int:%f, %f, %f\n", intersection1.x, intersection1.y, intersection1.z, intersection.x, intersection.y, intersection.z);
			glm::vec3 unit_int;
			// unit_int.x = intersection.x*intersection.x* (1/(scale.x*(scale.x)*9.0f + 0.0000001f));
			// unit_int.y = intersection.y*intersection.y* (1/(scale.y*(scale.y)*9.0f + 0.0000001f));
			// unit_int.z = intersection.z*intersection.z* (1/(scale.z*(scale.z)*9.0f + 0.0000001f));

			unit_int.x = intersection.x* (1/(sqrt(scales[collected_id[j]].x) + 0.0000001f));
			unit_int.y = intersection.y* (1/(sqrt(scales[collected_id[j]].y) + 0.0000001f));
			unit_int.z = intersection.z* (1/(sqrt(scales[collected_id[j]].z) + 0.0000001f));

			// unit_int.x = intersection.x* (1/(3.0f*scales[collected_id[j]].x + 0.0000001f));
			// unit_int.y = intersection.y* (1/(3.0f*scales[collected_id[j]].y + 0.0000001f));
			// unit_int.z = intersection.z* (1/(3.0f*scales[collected_id[j]].z + 0.0000001f));

			if (glm::length(unit_int) != 0)
				unit_int = unit_int / glm::length(unit_int);



			con_o.w = computeOpacityFromIntersection_f(collected_id[j], unit_int, texture_opacity, D);
			// if( length_squared == 0)
			// 	con_o.w = 0;
			





			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			// printf("int: %f,%f,%f, int1: %f,%f%f, alpha: %f, con_o.w:%f, G%f\n", intersection.x, intersection.y, intersection.z, intersection1.x, intersection1.y, intersection1.z, alpha, con_o.w, G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			glm::vec3 tmp;

			glm::vec3 rgb_out = computeColorFromD_f(collected_id[j], unit_int, texture, D);

			for (int ch = 0; ch < C; ch++)
			{
				// const float c = collected_colors[ch * BLOCK_SIZE + j];
				float c = 0;
				if (ch == 0 )
				{
					c = rgb_out.x;
				}
				else if(ch == 1)
				{
					c = rgb_out.y;
				}
				else
				{
					c = rgb_out.z;
				}
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
				if(ch == 0)
				{
					// printf("tmp.x: %f, %f\n", dL_dcolors[global_id * 3 + 0], dchannel_dcolor * dL_dchannel);
					// tmp.x = dL_dcolors[global_id * C + ch];
					tmp.x = dchannel_dcolor * dL_dchannel;
				}
				else if (ch == 1)
				{
					// tmp.y = dL_dcolors[global_id * C + ch];
					tmp.y = dchannel_dcolor * dL_dchannel;
				}
				else
				{
					// tmp.z = dL_dcolors[global_id * C + ch];
					tmp.z = dchannel_dcolor * dL_dchannel;
				}
				
				
			}
			// float3 mean = { means3D[collected_id[j]].x, means3D[collected_id[j]].y, means3D[collected_id[j]].z };
			// float3 ray = getRayVec_b(pixf, W, H, viewmatrix, projmatrix, *cam_pos, mean, rotations[collected_id[j]]);
			
			// float3 intersection = getIntersection_b(ray, mean, rotations[collected_id[j]], *cam_pos);
			computeColorFromD(global_id, unit_int, texture, tmp, dL_dtext, clamped, dL_dscale, rgb_out, D);
			// computeColorFromD1(global_id * 48, d, con_o, texture, tmp, dL_dtext, clamped);
			// the gradients are needed for every pixel of the Gaussian
			// computeColorFromD(int idx, const float* textures, dchannel_dcolor * dL_dchannel, glm::vec3* dL_dtext);
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			computeOpacityFromIntersection(global_id, unit_int, G * dL_dalpha, dL_dtext_opacity, clamped, dL_dscale, con_o.w, D);
			// atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);

		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* texture,
	const float* texture_opacity,
	const float* sig_out,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dtext,
	float* dL_dtext_opacity,
	const bool* clamped,
	const int D,
		const float* viewmatrix,
	const float* projmatrix,
	const float3* means3D,
	const glm::vec3* cam_pos,
	const glm::vec4* rotations,
	const glm::vec3* scales,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		texture,
		texture_opacity,
		sig_out,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dtext,
		dL_dtext_opacity,
		clamped,
		D,
		viewmatrix,
		projmatrix,
		means3D,
		cam_pos,
	rotations,
	scales,
	dL_dscale,
	dL_drot);
}