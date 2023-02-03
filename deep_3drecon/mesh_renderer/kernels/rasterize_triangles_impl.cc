// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cmath>

#include "rasterize_triangles_impl.h"

namespace tf_mesh_renderer {

namespace {

// Takes the minimum of a, b, and c, rounds down, and converts to an integer
// in the range [low, high].
inline int ClampedIntegerMin(float a, float b, float c, int low, int high) {
  return std::min(
      std::max(static_cast<int>(std::floor(std::min(std::min(a, b), c))), low),
      high);
}

// Takes the maximum of a, b, and c, rounds up, and converts to an integer
// in the range [low, high].
inline int ClampedIntegerMax(float a, float b, float c, int low, int high) {
  return std::min(
      std::max(static_cast<int>(std::ceil(std::max(std::max(a, b), c))), low),
      high);
}

// Computes a 3x3 matrix inverse without dividing by the determinant.
// Instead, makes an unnormalized matrix inverse with the correct sign
// by flipping the sign of the matrix if the determinant is negative.
// By leaving out determinant division, the rows of M^-1 only depend on two out
// of three of the columns of M; i.e., the first row of M^-1 only depends on the
// second and third columns of M, the second only depends on the first and
// third, etc. This means we can compute edge functions for two neighboring
// triangles independently and produce exactly the same numerical result up to
// the sign. This in turn means we can avoid cracks in rasterization without
// using fixed-point arithmetic.
// See http://mathworld.wolfram.com/MatrixInverse.html
void ComputeUnnormalizedMatrixInverse(const float a11, const float a12,
                                      const float a13, const float a21,
                                      const float a22, const float a23,
                                      const float a31, const float a32,
                                      const float a33, float m_inv[9]) {
  m_inv[0] = a22 * a33 - a32 * a23;
  m_inv[1] = a13 * a32 - a33 * a12;
  m_inv[2] = a12 * a23 - a22 * a13;
  m_inv[3] = a23 * a31 - a33 * a21;
  m_inv[4] = a11 * a33 - a31 * a13;
  m_inv[5] = a13 * a21 - a23 * a11;
  m_inv[6] = a21 * a32 - a31 * a22;
  m_inv[7] = a12 * a31 - a32 * a11;
  m_inv[8] = a11 * a22 - a21 * a12;

  // The first column of the unnormalized M^-1 contains intermediate values for
  // det(M).
  const float det = a11 * m_inv[0] + a12 * m_inv[3] + a13 * m_inv[6];

  // Transfer the sign of the determinant.
  if (det < 0.0f) {
    for (int i = 0; i < 9; ++i) {
      m_inv[i] = -m_inv[i];
    }
  }
}

// Computes the edge functions from M^-1 as described by Olano and Greer,
// "Triangle Scan Conversion using 2D Homogeneous Coordinates."
//
// This function combines equations (3) and (4). It first computes
// [a b c] = u_i * M^-1, where u_0 = [1 0 0], u_1 = [0 1 0], etc.,
// then computes edge_i = aX + bY + c
void ComputeEdgeFunctions(const float px, const float py, const float m_inv[9],
                          float values[3]) {
  for (int i = 0; i < 3; ++i) {
    const float a = m_inv[3 * i + 0];
    const float b = m_inv[3 * i + 1];
    const float c = m_inv[3 * i + 2];

    values[i] = a * px + b * py + c;
  }
}

// Determines whether the point p lies inside a front-facing triangle.
// Counts pixels exactly on an edge as inside the triangle, as long as the
// triangle is not degenerate. Degenerate (zero-area) triangles always fail the
// inside test.
bool PixelIsInsideTriangle(const float edge_values[3]) {
  // Check that the edge values are all non-negative and that at least one is
  // positive (triangle is non-degenerate).
  return (edge_values[0] >= 0 && edge_values[1] >= 0 && edge_values[2] >= 0) &&
         (edge_values[0] > 0 || edge_values[1] > 0 || edge_values[2] > 0);
}

}  // namespace

void RasterizeTrianglesImpl(const float* vertices, const int32* triangles,
                            int32 triangle_count, int32 image_width,
                            int32 image_height, int32* triangle_ids,
                            float* barycentric_coordinates, float* z_buffer) {
  const float half_image_width = 0.5 * image_width;
  const float half_image_height = 0.5 * image_height;
  float unnormalized_matrix_inverse[9];
  float b_over_w[3];

  for (int32 triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
    const int32 v0_x_id = 4 * triangles[3 * triangle_id];
    const int32 v1_x_id = 4 * triangles[3 * triangle_id + 1];
    const int32 v2_x_id = 4 * triangles[3 * triangle_id + 2];

    const float v0w = vertices[v0_x_id + 3];
    const float v1w = vertices[v1_x_id + 3];
    const float v2w = vertices[v2_x_id + 3];
    // Early exit: if all w < 0, triangle is entirely behind the eye.
    if (v0w < 0 && v1w < 0 && v2w < 0) {
      continue;
    }

    const float v0x = vertices[v0_x_id];
    const float v0y = vertices[v0_x_id + 1];
    const float v1x = vertices[v1_x_id];
    const float v1y = vertices[v1_x_id + 1];
    const float v2x = vertices[v2_x_id];
    const float v2y = vertices[v2_x_id + 1];

    ComputeUnnormalizedMatrixInverse(v0x, v1x, v2x, v0y, v1y, v2y, v0w, v1w,
                                     v2w, unnormalized_matrix_inverse);

    // Initialize the bounding box to the entire screen.
    int left = 0, right = image_width, bottom = 0, top = image_height;
    // If the triangle is entirely inside the screen, project the vertices to
    // pixel coordinates and find the triangle bounding box enlarged to the
    // nearest integer and clamped to the image boundaries.
    if (v0w > 0 && v1w > 0 && v2w > 0) {
      const float p0x = (v0x / v0w + 1.0) * half_image_width;
      const float p1x = (v1x / v1w + 1.0) * half_image_width;
      const float p2x = (v2x / v2w + 1.0) * half_image_width;
      const float p0y = (v0y / v0w + 1.0) * half_image_height;
      const float p1y = (v1y / v1w + 1.0) * half_image_height;
      const float p2y = (v2y / v2w + 1.0) * half_image_height;
      left = ClampedIntegerMin(p0x, p1x, p2x, 0, image_width);
      right = ClampedIntegerMax(p0x, p1x, p2x, 0, image_width);
      bottom = ClampedIntegerMin(p0y, p1y, p2y, 0, image_height);
      top = ClampedIntegerMax(p0y, p1y, p2y, 0, image_height);
    }

    // Iterate over each pixel in the bounding box.
    for (int iy = bottom; iy < top; ++iy) {
      for (int ix = left; ix < right; ++ix) {
        const float px = ((ix + 0.5) / half_image_width) - 1.0;
        const float py = ((iy + 0.5) / half_image_height) - 1.0;
        const int pixel_idx = iy * image_width + ix;

        ComputeEdgeFunctions(px, py, unnormalized_matrix_inverse, b_over_w);
        if (!PixelIsInsideTriangle(b_over_w)) {
          continue;
        }

        const float one_over_w = b_over_w[0] + b_over_w[1] + b_over_w[2];
        const float b0 = b_over_w[0] / one_over_w;
        const float b1 = b_over_w[1] / one_over_w;
        const float b2 = b_over_w[2] / one_over_w;

        const float v0z = vertices[v0_x_id + 2];
        const float v1z = vertices[v1_x_id + 2];
        const float v2z = vertices[v2_x_id + 2];
        // Since we computed an unnormalized w above, we need to recompute
        // a properly scaled clip-space w value and then divide clip-space z
        // by that.
        const float clip_z = b0 * v0z + b1 * v1z + b2 * v2z;
        const float clip_w = b0 * v0w + b1 * v1w + b2 * v2w;
        const float z = clip_z / clip_w;

        // Skip the pixel if it is farther than the current z-buffer pixel or
        // beyond the near or far clipping plane.
        if (z < -1.0 || z > 1.0 || z > z_buffer[pixel_idx]) {
          continue;
        }

        triangle_ids[pixel_idx] = triangle_id;
        z_buffer[pixel_idx] = z;
        barycentric_coordinates[3 * pixel_idx + 0] = b0;
        barycentric_coordinates[3 * pixel_idx + 1] = b1;
        barycentric_coordinates[3 * pixel_idx + 2] = b2;
      }
    }
  }
}

}  // namespace tf_mesh_renderer
