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
 *    Application description for Game of Life demo.
 *
 **************************************************************************/

#include "sim.hpp"

#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/ImageView.h>
#include <Magnum/Shaders/FlatGL.h>
#include <Magnum/Primitives/Square.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Trade/MeshData.h>

#include <sycl/sycl.hpp>
#include <thread>
#include <chrono>

constexpr Magnum::PixelFormat PIXELFORMAT{Magnum::PixelFormat::RGBA8Unorm};

class GameOfLifeApp : public Magnum::Platform::Application
{
  /// The window dimensions
  int m_width;
  int m_height;
  float m_zoom;

  /// Things to do with resizing the window. Have to be done to make it smooth.
  /// Wait this many frames before reinitializing after resize
  size_t RESIZE_TIMEOUT = 15;
  /// Counter for reinitializing after resize
  size_t m_resize_time = 0;
  /// Whether a resize has been executed
  size_t m_resized = false;
  /// Dimensions after the resize
  int m_resized_width;
  int m_resized_height;

  /// Whether the simulation is running or paused
  bool m_paused = false;

  /// The simulation
  GameOfLifeSim m_sim;

  /// The texture to display the simulation on
  Magnum::GL::Texture2D m_tex;

  // Mesh and shader to draw the texture
  Magnum::GL::Mesh m_mesh;
  Magnum::Shaders::FlatGL2D m_shader;

 public:
  GameOfLifeApp(const Arguments& arguments)
  : Magnum::Platform::Application{
      arguments,
      Configuration{}.setTitle("Codeplay Game of Life Demo")
                     .addWindowFlags(Configuration::WindowFlag::Resizable),
      GLConfiguration{}.setFlags(GLConfiguration::Flag::QuietLog)},
    m_width(windowSize().x()),
    m_height(windowSize().y()),
    m_zoom(1),
    m_resized_width{m_width},
    m_resized_height{m_height},
    m_sim(m_width, m_height),
    m_mesh{Magnum::MeshTools::compile(Magnum::Primitives::squareSolid(
      Magnum::Primitives::SquareFlag::TextureCoordinates))},
    m_shader{Magnum::Shaders::FlatGL2D::Configuration{}.setFlags(
      Magnum::Shaders::FlatGL2D::Flag::Textured |
      Magnum::Shaders::FlatGL2D::Flag::TextureTransformation)} {

    m_tex.setWrapping(Magnum::GL::SamplerWrapping::ClampToEdge)
         .setMagnificationFilter(Magnum::GL::SamplerFilter::Nearest)
         .setMinificationFilter(Magnum::GL::SamplerFilter::Nearest)
         .setStorage(1, Magnum::GL::textureFormat(PIXELFORMAT), {m_width,m_height});
    m_shader.bindTexture(m_tex);
  }

  void tickEvent() override {
    if (m_resized) {
      if (m_resize_time++ >= RESIZE_TIMEOUT) {
        m_width = m_resized_width / m_zoom;
        m_height = m_resized_height / m_zoom;

        m_sim = GameOfLifeSim(m_width, m_height);
        // Reinitializes image to new size
        m_tex = Magnum::GL::Texture2D{};
        m_tex.setWrapping(Magnum::GL::SamplerWrapping::ClampToEdge)
             .setMagnificationFilter(Magnum::GL::SamplerFilter::Nearest)
             .setMinificationFilter(Magnum::GL::SamplerFilter::Nearest)
             .setStorage(1, Magnum::GL::textureFormat(PIXELFORMAT), {m_width,m_height});
        m_shader.bindTexture(m_tex);

        m_resized = false;
        m_resize_time = 0;
      }
    }

    if (!m_paused) {
      m_sim.step();
      // 60 FPS
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
  }

  void drawEvent() override {
    Magnum::GL::defaultFramebuffer.clear(
        Magnum::GL::FramebufferClear::Color |
        Magnum::GL::FramebufferClear::Depth);

    m_sim.with_img(
        // Gets called with image data
        [&](sycl::uchar4 const* data) {
          Magnum::ImageView2D img{
            PIXELFORMAT, {m_width, m_height},
            Corrade::Containers::ArrayView{
              reinterpret_cast<const char*>(data),
              m_width*m_height*Magnum::pixelFormatSize(PIXELFORMAT)}};
          m_tex.setSubImage(0, {0,0}, img);
        });

    m_shader.draw(m_mesh);
    redraw();
    swapBuffers();
  }

  void handleMouse(size_t mouse_x, size_t mouse_y) {
    // Obtain coordinates within the simulation dimensions
    size_t x = static_cast<float>(mouse_x) /
               static_cast<float>(windowSize().x()) *
               static_cast<float>(m_width);
    size_t y = static_cast<float>(mouse_y) /
               static_cast<float>(windowSize().y()) *
               static_cast<float>(m_height);
    // Invert Y
    y = m_height - y;
    if (x < m_width && y < m_height) {
      // Set cell at mouse position to alive
      m_sim.add_click(x, y + 1, CellState::LIVE);
      m_sim.add_click(x + 1, y, CellState::LIVE);
      m_sim.add_click(x, y - 1, CellState::LIVE);
      m_sim.add_click(x - 1, y - 1, CellState::LIVE);      
    }    
  }

  void mousePressEvent(MouseEvent& event) override {
    handleMouse(event.position().x(), event.position().y());
  }

  void mouseMoveEvent(MouseMoveEvent& event) override {
    // Drag only if left button clicked
    if ((event.buttons() & MouseMoveEvent::Button::Left)) {
      handleMouse(event.position().x(), event.position().y());
    }
  }

  void mouseScrollEvent(MouseScrollEvent& event) override {
    auto prevZoom = m_zoom;
    auto inc = event.offset().y();
    if (inc > 0) {
      // Zoom in on wheel up
      m_zoom *= 2.0f * inc;
    } else {
      // Zoom out on wheel down
      m_zoom /= -2.0f * inc;
    }
    // Don't zoom out further than one cell per pixel
    m_zoom = sycl::clamp(m_zoom, 1.0f, 64.0f);

    // Restart after zooming
    if (m_zoom!=prevZoom) {m_resized = true;}
  }

  void keyPressEvent(KeyEvent& event) override {
    // (Un)pause on SPACE
    if (event.key() == KeyEvent::Key::Space) {
      m_paused = !m_paused;
    }
  }

  void viewportEvent(ViewportEvent& event) override {
    m_resized_width = event.windowSize().x();
    m_resized_height = event.windowSize().y();
    m_resized = true;
    Magnum::GL::defaultFramebuffer.setViewport({{0,0}, event.framebufferSize()});
  }
};

MAGNUM_APPLICATION_MAIN(GameOfLifeApp)
