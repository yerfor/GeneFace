# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable 3-D rendering of a triangle mesh."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import camera_utils
from . import rasterize_triangles
# import camera_utils
# import rasterize_triangles


def phong_shader(normals,
                 alphas,
                 pixel_positions,
                 light_positions,
                 light_intensities,
                 diffuse_colors=None,
                 camera_position=None,
                 specular_colors=None,
                 shininess_coefficients=None,
                 ambient_color=None):
  """Computes pixelwise lighting from rasterized buffers with the Phong model.

  Args:
    normals: a 4D float32 tensor with shape [batch_size, image_height,
        image_width, 3]. The inner dimension is the world space XYZ normal for
        the corresponding pixel. Should be already normalized.
    alphas: a 3D float32 tensor with shape [batch_size, image_height,
        image_width]. The inner dimension is the alpha value (transparency)
        for the corresponding pixel.
    pixel_positions: a 4D float32 tensor with shape [batch_size, image_height,
        image_width, 3]. The inner dimension is the world space XYZ position for
        the corresponding pixel.
    light_positions: a 3D tensor with shape [batch_size, light_count, 3]. The
        XYZ position of each light in the scene. In the same coordinate space as
        pixel_positions.
    light_intensities: a 3D tensor with shape [batch_size, light_count, 3]. The
        RGB intensity values for each light. Intensities may be above one.
    diffuse_colors: a 4D float32 tensor with shape [batch_size, image_height,
        image_width, 3]. The inner dimension is the diffuse RGB coefficients at
        a pixel in the range [0, 1].
    camera_position: a 1D tensor with shape [batch_size, 3]. The XYZ camera
        position in the scene. If supplied, specular reflections will be
        computed. If not supplied, specular_colors and shininess_coefficients
        are expected to be None. In the same coordinate space as
        pixel_positions.
    specular_colors: a 4D float32 tensor with shape [batch_size, image_height,
        image_width, 3]. The inner dimension is the specular RGB coefficients at
        a pixel in the range [0, 1]. If None, assumed to be tf.zeros()
    shininess_coefficients: A 3D float32 tensor that is broadcasted to shape
        [batch_size, image_height, image_width]. The inner dimension is the
        shininess coefficient for the object at a pixel. Dimensions that are
        constant can be given length 1, so [batch_size, 1, 1] and [1, 1, 1] are
        also valid input shapes.
    ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
        color, which is added to each pixel before tone mapping. If None, it is
        assumed to be tf.zeros().
  Returns:
    A 4D float32 tensor of shape [batch_size, image_height, image_width, 4]
    containing the lit RGBA color values for each image at each pixel. Colors
    are in the range [0,1].

  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  batch_size, image_height, image_width = [s.value for s in normals.shape[:-1]]
  light_count = light_positions.shape[1].value
  pixel_count = image_height * image_width
  # Reshape all values to easily do pixelwise computations:
  normals = tf.reshape(normals, [batch_size, -1, 3])
  alphas = tf.reshape(alphas, [batch_size, -1, 1])
  diffuse_colors = tf.reshape(diffuse_colors, [batch_size, -1, 3])
  if camera_position is not None:
    specular_colors = tf.reshape(specular_colors, [batch_size, -1, 3])

  # Ambient component
  output_colors = tf.zeros([batch_size, image_height * image_width, 3])
  if ambient_color is not None:
    ambient_reshaped = tf.expand_dims(ambient_color, axis=1)
    output_colors = tf.add(output_colors, ambient_reshaped * diffuse_colors)

  # Diffuse component
  pixel_positions = tf.reshape(pixel_positions, [batch_size, -1, 3])
  per_light_pixel_positions = tf.stack(
      [pixel_positions] * light_count,
      axis=1)  # [batch_size, light_count, pixel_count, 3]
  directions_to_lights = tf.nn.l2_normalize(
      tf.expand_dims(light_positions, axis=2) - per_light_pixel_positions,
      axis=3)  # [batch_size, light_count, pixel_count, 3]
  # The specular component should only contribute when the light and normal
  # face one another (i.e. the dot product is nonnegative):
  normals_dot_lights = tf.clip_by_value(
      tf.reduce_sum(
          tf.expand_dims(normals, axis=1) * directions_to_lights, axis=3), 0.0,
      1.0)  # [batch_size, light_count, pixel_count]
  diffuse_output = tf.expand_dims(
      diffuse_colors, axis=1) * tf.expand_dims(
          normals_dot_lights, axis=3) * tf.expand_dims(
              light_intensities, axis=2)
  diffuse_output = tf.reduce_sum(
      diffuse_output, axis=1)  # [batch_size, pixel_count, 3]
  output_colors = tf.add(output_colors, diffuse_output)

  # Specular component
  if camera_position is not None:
    camera_position = tf.reshape(camera_position, [batch_size, 1, 3])
    mirror_reflection_direction = tf.nn.l2_normalize(
        2.0 * tf.expand_dims(normals_dot_lights, axis=3) * tf.expand_dims(
            normals, axis=1) - directions_to_lights,
        dim=3)
    direction_to_camera = tf.nn.l2_normalize(
        camera_position - pixel_positions, dim=2)
    reflection_direction_dot_camera_direction = tf.reduce_sum(
        tf.expand_dims(direction_to_camera, axis=1) *
        mirror_reflection_direction,
        axis=3)
    # The specular component should only contribute when the reflection is
    # external:
    reflection_direction_dot_camera_direction = tf.clip_by_value(
        tf.nn.l2_normalize(reflection_direction_dot_camera_direction, dim=2),
        0.0, 1.0)
    # The specular component should also only contribute when the diffuse
    # component contributes:
    reflection_direction_dot_camera_direction = tf.where(
        normals_dot_lights != 0.0, reflection_direction_dot_camera_direction,
        tf.zeros_like(
            reflection_direction_dot_camera_direction, dtype=tf.float32))
    # Reshape to support broadcasting the shininess coefficient, which rarely
    # varies per-vertex:
    reflection_direction_dot_camera_direction = tf.reshape(
        reflection_direction_dot_camera_direction,
        [batch_size, light_count, image_height, image_width])
    shininess_coefficients = tf.expand_dims(shininess_coefficients, axis=1)
    specularity = tf.reshape(
        tf.pow(reflection_direction_dot_camera_direction,
               shininess_coefficients),
        [batch_size, light_count, pixel_count, 1])
    specular_output = tf.expand_dims(
        specular_colors, axis=1) * specularity * tf.expand_dims(
            light_intensities, axis=2)
    specular_output = tf.reduce_sum(specular_output, axis=1)
    output_colors = tf.add(output_colors, specular_output)
  rgb_images = tf.reshape(output_colors,
                          [batch_size, image_height, image_width, 3])
  alpha_images = tf.reshape(alphas, [batch_size, image_height, image_width, 1])
  valid_rgb_values = tf.concat(3 * [alpha_images > 0.5], axis=3)
  rgb_images = tf.where(valid_rgb_values, rgb_images,
                        tf.zeros_like(rgb_images, dtype=tf.float32))
  return tf.reverse(tf.concat([rgb_images, alpha_images], axis=3), axis=[1])


def tone_mapper(image, gamma):
  """Applies gamma correction to the input image.

  Tone maps the input image batch in order to make scenes with a high dynamic
  range viewable. The gamma correction factor is computed separately per image,
  but is shared between all provided channels. The exact function computed is:

  image_out = A*image_in^gamma, where A is an image-wide constant computed so
  that the maximum image value is approximately 1. The correction is applied
  to all channels.

  Args:
    image: 4-D float32 tensor with shape [batch_size, image_height,
        image_width, channel_count]. The batch of images to tone map.
    gamma: 0-D float32 nonnegative tensor. Values of gamma below one compress
        relative contrast in the image, and values above one increase it. A
        value of 1 is equivalent to scaling the image to have a maximum value
        of 1.
  Returns:
    4-D float32 tensor with shape [batch_size, image_height, image_width,
    channel_count]. Contains the gamma-corrected images, clipped to the range
    [0, 1].
  """
  batch_size = image.shape[0].value
  corrected_image = tf.pow(image, gamma)
  image_max = tf.reduce_max(
      tf.reshape(corrected_image, [batch_size, -1]), axis=1)
  scaled_image = tf.divide(corrected_image,
                           tf.reshape(image_max, [batch_size, 1, 1, 1]))
  return tf.clip_by_value(scaled_image, 0.0, 1.0)


def mesh_renderer(vertices,
                  triangles,
                  normals,
                  diffuse_colors,
                  camera_position,
                  camera_lookat,
                  camera_up,
                  light_positions,
                  light_intensities,
                  image_width,
                  image_height,
                  specular_colors=None,
                  shininess_coefficients=None,
                  ambient_color=None,
                  fov_y=40.0,
                  near_clip=0.01,
                  far_clip=10.0):
  """Renders an input scene using phong shading, and returns an output image.

  Args:
    vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
        should contain vertex indices describing a triangle such that the
        triangle's normal points toward the viewer if the forward order of the
        triplet defines a clockwise winding of the vertices. Gradients with
        respect to this tensor are not available.
    normals: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is the xyz vertex normal for its corresponding vertex. Each
        vector is assumed to be already normalized.
    diffuse_colors: 3-D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB diffuse reflection in the range [0,1] for
        each vertex.
    camera_position: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
        shape [3] specifying the XYZ world space camera position.
    camera_lookat: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
        shape [3] containing an XYZ point along the center of the camera's gaze.
    camera_up: 2-D tensor with shape [batch_size, 3] or 1-D tensor with shape
        [3] containing the up direction for the camera. The camera will have no
        tilt with respect to this direction.
    light_positions: a 3-D tensor with shape [batch_size, light_count, 3]. The
        XYZ position of each light in the scene. In the same coordinate space as
        pixel_positions.
    light_intensities: a 3-D tensor with shape [batch_size, light_count, 3]. The
        RGB intensity values for each light. Intensities may be above one.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    specular_colors: 3-D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB specular reflection in the range [0, 1] for
        each vertex.  If supplied, specular reflections will be computed, and
        both specular_colors and shininess_coefficients are expected.
    shininess_coefficients: a 0D-2D float32 tensor with maximum shape
       [batch_size, vertex_count]. The phong shininess coefficient of each
       vertex. A 0D tensor or float gives a constant shininess coefficient
       across all batches and images. A 1D tensor must have shape [batch_size],
       and a single shininess coefficient per image is used.
    ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
        color, which is added to each pixel in the scene. If None, it is
        assumed to be black.
    fov_y: float, 0D tensor, or 1D tensor with shape [batch_size] specifying
        desired output image y field of view in degrees.
    near_clip: float, 0D tensor, or 1D tensor with shape [batch_size] specifying
        near clipping plane distance.
    far_clip: float, 0D tensor, or 1D tensor with shape [batch_size] specifying
        far clipping plane distance.

  Returns:
    A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
    containing the lit RGBA color values for each image at each pixel. RGB
    colors are the intensity values before tonemapping and can be in the range
    [0, infinity]. Clipping to the range [0,1] with tf.clip_by_value is likely
    reasonable for both viewing and training most scenes. More complex scenes
    with multiple lights should tone map color values for display only. One
    simple tonemapping approach is to rescale color values as x/(1+x); gamma
    compression is another common techinque. Alpha values are zero for
    background pixels and near one for mesh pixels.
  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  if len(vertices.shape) != 3:
    raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
  batch_size = vertices.shape[0].value
  if len(normals.shape) != 3:
    raise ValueError('Normals must have shape [batch_size, vertex_count, 3].')
  if len(light_positions.shape) != 3:
    raise ValueError(
        'Light_positions must have shape [batch_size, light_count, 3].')
  if len(light_intensities.shape) != 3:
    raise ValueError(
        'Light_intensities must have shape [batch_size, light_count, 3].')
  if len(diffuse_colors.shape) != 3:
    raise ValueError(
        'vertex_diffuse_colors must have shape [batch_size, vertex_count, 3].')
  if (ambient_color is not None and
      ambient_color.get_shape().as_list() != [batch_size, 3]):
    raise ValueError('Ambient_color must have shape [batch_size, 3].')
  if camera_position.get_shape().as_list() == [3]:
    camera_position = tf.tile(
        tf.expand_dims(camera_position, axis=0), [batch_size, 1])
  elif camera_position.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_position must have shape [batch_size, 3]')
  if camera_lookat.get_shape().as_list() == [3]:
    camera_lookat = tf.tile(
        tf.expand_dims(camera_lookat, axis=0), [batch_size, 1])
  elif camera_lookat.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_lookat must have shape [batch_size, 3]')
  if camera_up.get_shape().as_list() == [3]:
    camera_up = tf.tile(tf.expand_dims(camera_up, axis=0), [batch_size, 1])
  elif camera_up.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_up must have shape [batch_size, 3]')
  if isinstance(fov_y, float):
    fov_y = tf.constant(batch_size * [fov_y], dtype=tf.float32)
  elif not fov_y.get_shape().as_list():
    fov_y = tf.tile(tf.expand_dims(fov_y, 0), [batch_size])
  elif fov_y.get_shape().as_list() != [batch_size]:
    raise ValueError('Fov_y must be a float, a 0D tensor, or a 1D tensor with'
                     'shape [batch_size]')
  if isinstance(near_clip, float):
    near_clip = tf.constant(batch_size * [near_clip], dtype=tf.float32)
  elif not near_clip.get_shape().as_list():
    near_clip = tf.tile(tf.expand_dims(near_clip, 0), [batch_size])
  elif near_clip.get_shape().as_list() != [batch_size]:
    raise ValueError('Near_clip must be a float, a 0D tensor, or a 1D tensor'
                     'with shape [batch_size]')
  if isinstance(far_clip, float):
    far_clip = tf.constant(batch_size * [far_clip], dtype=tf.float32)
  elif not far_clip.get_shape().as_list():
    far_clip = tf.tile(tf.expand_dims(far_clip, 0), [batch_size])
  elif far_clip.get_shape().as_list() != [batch_size]:
    raise ValueError('Far_clip must be a float, a 0D tensor, or a 1D tensor'
                     'with shape [batch_size]')
  if specular_colors is not None and shininess_coefficients is None:
    raise ValueError(
        'Specular colors were supplied without shininess coefficients.')
  if shininess_coefficients is not None and specular_colors is None:
    raise ValueError(
        'Shininess coefficients were supplied without specular colors.')
  if specular_colors is not None:
    # Since a 0-D float32 tensor is accepted, also accept a float.
    if isinstance(shininess_coefficients, float):
      shininess_coefficients = tf.constant(
          shininess_coefficients, dtype=tf.float32)
    if len(specular_colors.shape) != 3:
      raise ValueError('The specular colors must have shape [batch_size, '
                       'vertex_count, 3].')
    if len(shininess_coefficients.shape) > 2:
      raise ValueError('The shininess coefficients must have shape at most'
                       '[batch_size, vertex_count].')
    # If we don't have per-vertex coefficients, we can just reshape the
    # input shininess to broadcast later, rather than interpolating an
    # additional vertex attribute:
    if len(shininess_coefficients.shape) < 2:
      vertex_attributes = tf.concat(
          [normals, vertices, diffuse_colors, specular_colors], axis=2)
    else:
      vertex_attributes = tf.concat(
          [
              normals, vertices, diffuse_colors, specular_colors,
              tf.expand_dims(shininess_coefficients, axis=2)
          ],
          axis=2)
  else:
    vertex_attributes = tf.concat([normals, vertices, diffuse_colors], axis=2)

  camera_matrices = camera_utils.look_at(camera_position, camera_lookat,
                                         camera_up)

  perspective_transforms = camera_utils.perspective(image_width / image_height,
                                                    fov_y, near_clip, far_clip)

  clip_space_transforms = tf.matmul(perspective_transforms, camera_matrices)

  pixel_attributes = rasterize_triangles.rasterize(
      vertices, vertex_attributes, triangles, clip_space_transforms,
      image_width, image_height, [-1] * vertex_attributes.shape[2].value)

  # Extract the interpolated vertex attributes from the pixel buffer and
  # supply them to the shader:
  pixel_normals = tf.nn.l2_normalize(pixel_attributes[:, :, :, 0:3], axis=3)
  pixel_positions = pixel_attributes[:, :, :, 3:6]
  diffuse_colors = pixel_attributes[:, :, :, 6:9]
  if specular_colors is not None:
    specular_colors = pixel_attributes[:, :, :, 9:12]
    # Retrieve the interpolated shininess coefficients if necessary, or just
    # reshape our input for broadcasting:
    if len(shininess_coefficients.shape) == 2:
      shininess_coefficients = pixel_attributes[:, :, :, 12]
    else:
      shininess_coefficients = tf.reshape(shininess_coefficients, [-1, 1, 1])

  pixel_mask = tf.cast(tf.reduce_any(diffuse_colors >= 0, axis=3), tf.float32)

  renders = phong_shader(
      normals=pixel_normals,
      alphas=pixel_mask,
      pixel_positions=pixel_positions,
      light_positions=light_positions,
      light_intensities=light_intensities,
      diffuse_colors=diffuse_colors,
      camera_position=camera_position if specular_colors is not None else None,
      specular_colors=specular_colors,
      shininess_coefficients=shininess_coefficients,
      ambient_color=ambient_color)
  return renders

def mesh_uv(vertices,
                  triangles,
                  normals,
                  uv,
                  camera_position,
                  camera_lookat,
                  camera_up,
                  image_width,
                  image_height,
                  fov_y=40.0,
                  near_clip=0.01,
                  far_clip=10.0):
  
  if len(vertices.shape) != 3:
    raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
  batch_size = vertices.shape[0].value
  if len(normals.shape) != 3:
    raise ValueError('Normals must have shape [batch_size, vertex_count, 3].')
  if len(uv.shape) != 3:
    raise ValueError(
        'vertex_diffuse_colors must have shape [batch_size, vertex_count, 3].')
  if camera_position.get_shape().as_list() == [3]:
    camera_position = tf.tile(
        tf.expand_dims(camera_position, axis=0), [batch_size, 1])
  elif camera_position.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_position must have shape [batch_size, 3]')
  if camera_lookat.get_shape().as_list() == [3]:
    camera_lookat = tf.tile(
        tf.expand_dims(camera_lookat, axis=0), [batch_size, 1])
  elif camera_lookat.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_lookat must have shape [batch_size, 3]')
  if camera_up.get_shape().as_list() == [3]:
    camera_up = tf.tile(tf.expand_dims(camera_up, axis=0), [batch_size, 1])
  elif camera_up.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_up must have shape [batch_size, 3]')
  if isinstance(fov_y, float):
    fov_y = tf.constant(batch_size * [fov_y], dtype=tf.float32)
  elif not fov_y.get_shape().as_list():
    fov_y = tf.tile(tf.expand_dims(fov_y, 0), [batch_size])
  elif fov_y.get_shape().as_list() != [batch_size]:
    raise ValueError('Fov_y must be a float, a 0D tensor, or a 1D tensor with'
                     'shape [batch_size]')
  if isinstance(near_clip, float):
    near_clip = tf.constant(batch_size * [near_clip], dtype=tf.float32)
  elif not near_clip.get_shape().as_list():
    near_clip = tf.tile(tf.expand_dims(near_clip, 0), [batch_size])
  elif near_clip.get_shape().as_list() != [batch_size]:
    raise ValueError('Near_clip must be a float, a 0D tensor, or a 1D tensor'
                     'with shape [batch_size]')
  if isinstance(far_clip, float):
    far_clip = tf.constant(batch_size * [far_clip], dtype=tf.float32)
  elif not far_clip.get_shape().as_list():
    far_clip = tf.tile(tf.expand_dims(far_clip, 0), [batch_size])
  elif far_clip.get_shape().as_list() != [batch_size]:
    raise ValueError('Far_clip must be a float, a 0D tensor, or a 1D tensor'
                     'with shape [batch_size]')
  
  vertex_attributes = tf.concat([uv], axis=2)

  camera_matrices = camera_utils.look_at(camera_position, camera_lookat,
                                         camera_up)

  perspective_transforms = camera_utils.perspective(image_width / image_height,
                                                    fov_y, near_clip, far_clip)

  clip_space_transforms = tf.matmul(perspective_transforms, camera_matrices)

  pixel_attributes = rasterize_triangles.rasterize(
      vertices, vertex_attributes, triangles, clip_space_transforms,
      image_width, image_height, [-1] * vertex_attributes.shape[2].value)

  
  uv_images = pixel_attributes
  return uv_images

def rasterize_texture(
  uvcoords,
  colors,
  triangles,
  image_width,
  image_height,
):
  vertex_attributes = tf.concat([colors], axis=2)
  pixel_attributes = rasterize_triangles.rasterize_clip_space(uvcoords, vertex_attributes, triangles, image_width, image_height, [0] * vertex_attributes.shape[2].value)
  return pixel_attributes

def clip_vertices(vertices,
                  camera_position,
                  camera_lookat,
                  camera_up,
                  image_width,
                  image_height,
                  fov_y=40.0,
                  near_clip=0.01,
                  far_clip=10.0):
  if len(vertices.shape) != 3:
    raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
  batch_size = vertices.shape[0].value
  if camera_position.get_shape().as_list() == [3]:
    camera_position = tf.tile(
        tf.expand_dims(camera_position, axis=0), [batch_size, 1])
  elif camera_position.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_position must have shape [batch_size, 3]')
  if camera_lookat.get_shape().as_list() == [3]:
    camera_lookat = tf.tile(
        tf.expand_dims(camera_lookat, axis=0), [batch_size, 1])
  elif camera_lookat.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_lookat must have shape [batch_size, 3]')
  if camera_up.get_shape().as_list() == [3]:
    camera_up = tf.tile(tf.expand_dims(camera_up, axis=0), [batch_size, 1])
  elif camera_up.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_up must have shape [batch_size, 3]')
  if isinstance(fov_y, float):
    fov_y = tf.constant(batch_size * [fov_y], dtype=tf.float32)
  elif not fov_y.get_shape().as_list():
    fov_y = tf.tile(tf.expand_dims(fov_y, 0), [batch_size])
  elif fov_y.get_shape().as_list() != [batch_size]:
    raise ValueError('Fov_y must be a float, a 0D tensor, or a 1D tensor with'
                     'shape [batch_size]')
  if isinstance(near_clip, float):
    near_clip = tf.constant(batch_size * [near_clip], dtype=tf.float32)
  elif not near_clip.get_shape().as_list():
    near_clip = tf.tile(tf.expand_dims(near_clip, 0), [batch_size])
  elif near_clip.get_shape().as_list() != [batch_size]:
    raise ValueError('Near_clip must be a float, a 0D tensor, or a 1D tensor'
                     'with shape [batch_size]')
  if isinstance(far_clip, float):
    far_clip = tf.constant(batch_size * [far_clip], dtype=tf.float32)
  elif not far_clip.get_shape().as_list():
    far_clip = tf.tile(tf.expand_dims(far_clip, 0), [batch_size])
  elif far_clip.get_shape().as_list() != [batch_size]:
    raise ValueError('Far_clip must be a float, a 0D tensor, or a 1D tensor'
                     'with shape [batch_size]')

  camera_matrices = camera_utils.look_at(camera_position, camera_lookat,
                                         camera_up)

  perspective_transforms = camera_utils.perspective(image_width / image_height,
                                                    fov_y, near_clip, far_clip)

  clip_space_transforms = tf.matmul(perspective_transforms, camera_matrices)

  clip_space_vertices = camera_utils.transform_homogeneous(
      clip_space_transforms, vertices)
  return clip_space_vertices
