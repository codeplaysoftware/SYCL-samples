/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Description:
 *    SYCL kernel for Mandelbrot demo.
 *
 **************************************************************************/

#pragma once

#include <sycl/sycl.hpp>

template <typename num_t>
class CalcKernel;

/* Computes an image representing the Mandelbrot set on the complex
 * plane at a given zoom level. */
class MandelbrotCalculator {
  // Dimensions of the image to be calculated
  size_t const m_width;
  size_t const m_height;

  // Accelerated SYCL queue and storage for image data
  sycl::queue m_q;
  sycl::buffer<sycl::uchar4, 2> m_img;

  // Boundaries on the part of the complex plane which we want to view
  double m_minx = -2;
  double m_maxx = 1;
  double m_miny = -1;
  double m_maxy = 1;

  bool m_supports_doubles{true};

 public:
  MandelbrotCalculator(size_t width, size_t height)
      : m_width(width),
        m_height(height),
        m_q(sycl::default_selector_v,
            [&](sycl::exception_list el) {
              for (auto const& e : el) {
                try {
                  std::rethrow_exception(e);
                } catch (std::exception const& e) {
                  std::cout << "SYCL exception caught:\n:" << e.what()
                            << std::endl;
                }
              }
            }),
        // These are flipped since OpenGL expects column-major order for
        // textures
        m_img(sycl::range<2>(height, width)),
        // If the vector returned by get_info<double_fp_config> is length 0
        // doubles are not supported by the SYCL device.
        m_supports_doubles(m_q.get_device()
                               .get_info<sycl::info::device::double_fp_config>()
                               .size() != 0) {}

  // Set the boundaries of the viewable region. X is Re, Y is Im.
  void set_bounds(double min_x, double max_x, double min_y, double max_y) {
    m_minx = min_x;
    m_maxx = max_x;
    m_miny = min_y;
    m_maxy = max_y;
  }

  bool supports_doubles() const { return m_supports_doubles; }

  template <typename num_t>
  void calc();

  // Calls the function with the underlying image memory.
  template <typename Func>
  void with_data(Func&& func) {
    auto acc = m_img.get_host_access(sycl::read_only);

    func(acc.get_pointer());
  }

 private:
  template <typename num_t>
  void internal_calc() {
    m_q.submit([&](sycl::handler& cgh) {
      auto img_acc = m_img.get_access(cgh, sycl::write_only);

      static constexpr size_t MAX_ITERS = 500;
      // Anything above this number is assumed divergent. To do less
      // computation, this is the _square_ of the maximum absolute value
      // of a non-divergent number
      static constexpr num_t DIVERGENCE_LIMIT = num_t(256);

      // Calculates how many iterations does it take to diverge? MAX_ITERS if in
      // Mandelbrot set
      const auto how_mandel = [](num_t re, num_t im) -> num_t {
        num_t z_re = 0;
        num_t z_im = 0;
        num_t abs_sq = 0;

        for (size_t i = 0; i < MAX_ITERS; i++) {
          num_t z_re2 = z_re * z_re - z_im * z_im + re;
          z_im = num_t(2) * z_re * z_im + im;
          z_re = z_re2;

          abs_sq = z_re * z_re + z_im * z_im;

          // Branching here isn't ideal, but it's the simplest
          if (abs_sq >= DIVERGENCE_LIMIT) {
            num_t log_zn = sycl::log(abs_sq) / num_t(2);
            num_t nu =
                sycl::log(log_zn / sycl::log(num_t(2))) / sycl::log(num_t(2));
            return num_t(i) + num_t(1) - nu;
          }
        }

        return num_t(1);
      };

      // Dummy variable copies to avoid capturing `this` in kernel lambda
      size_t width = m_width;
      size_t height = m_height;
      num_t minx = m_minx;
      num_t maxx = m_maxx;
      num_t miny = m_miny;
      num_t maxy = m_maxy;

      cgh.parallel_for<CalcKernel<num_t>>(
          sycl::range<2>(m_height, m_width), [=](sycl::item<2> item) {
            // Obtain normalized coords [0, 1]
            num_t x = num_t(item.get_id(1)) / num_t(width);
            num_t y = num_t(item.get_id(0)) / num_t(height);

            // Put them within desired bounds
            x *= (maxx - minx);
            x += minx;

            y *= (maxy - miny);
            y += miny;

            // Calculate sequence divergence
            num_t mandelness = how_mandel(x, y);

            // Map to two colors in the palette
            const std::array<sycl::vec<num_t, 4>, 16> COLORS = {{
                {66, 30, 15, 255},
                {25, 7, 26, 255},
                {9, 1, 47, 255},
                {4, 4, 73, 255},
                {0, 7, 100, 255},
                {12, 44, 138, 255},
                {24, 82, 177, 255},
                {57, 125, 209, 255},
                {134, 181, 229, 255},
                {211, 236, 248, 255},
                {241, 233, 191, 255},
                {248, 201, 95, 255},
                {255, 170, 0, 255},
                {204, 128, 0, 255},
                {153, 87, 0, 255},
                {106, 52, 3, 255},
            }};

            auto col_a = COLORS[size_t(mandelness) % COLORS.size()];
            auto col_b = COLORS[(size_t(mandelness) + 1) % COLORS.size()];

            // fract(a) = a - floor(a)
            auto fract = mandelness - num_t(size_t(mandelness));

            // Linearly interpolate between the colors using the fractional part
            // of 'mandelness' to get smooth transitions
            auto col = col_a * (num_t(1) - fract) + col_b * fract;

            // Store color in image
            img_acc[item] = {col.x(), col.y(), col.z(), col.w()};
          });
    });
  }
};
