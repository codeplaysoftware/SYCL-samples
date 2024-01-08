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
 *    Application description for NBody demo.
 *
 **************************************************************************/

#include <Corrade/PluginManager/Manager.h>
#include <Corrade/Utility/Resource.h>
#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Version.h>
#include <Magnum/ImageView.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Vector.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData.h>

#include <Magnum/ImGuiIntegration/Context.hpp>
#include <chrono>
#include <fstream>

#include <sycl/sycl.hpp>

#include "sim.hpp"

using num_t = float;
constexpr num_t PI{3.141592653589793238462643383279502884197169399};

class NBodyShader : public Magnum::GL::AbstractShaderProgram {
 public:
  using Position = Magnum::GL::Attribute<0, Magnum::Vector3>;
  using Velocity = Magnum::GL::Attribute<1, Magnum::Vector3>;
  enum : unsigned int { ColourOutput = 0 };
  explicit NBodyShader() {
    /* Load shader sources */
    Magnum::GL::Shader vert{Magnum::GL::Version::GL330,
                            Magnum::GL::Shader::Type::Vertex};
    Magnum::GL::Shader frag{Magnum::GL::Version::GL330,
                            Magnum::GL::Shader::Type::Fragment};
    const Corrade::Utility::Resource rs{"nbody-data"};
    vert.addSource(rs.getString("NBodyShader.vert"));
    frag.addSource(rs.getString("NBodyShader.frag"));

    /* Compile them */
    CORRADE_INTERNAL_ASSERT_OUTPUT(vert.compile() && frag.compile());

    /* Attach the shaders */
    attachShaders({vert, frag});

    bindAttributeLocation(Position::Location, "ciPosition");
    bindAttributeLocation(Velocity::Location, "ciVelocity");
    bindFragmentDataLocation(ColourOutput, "oColor");

    /* Link the program together */
    CORRADE_INTERNAL_ASSERT_OUTPUT(link());
  }

  NBodyShader& setView(
      Corrade::Containers::ArrayView<
          const Magnum::Math::RectangularMatrix<4, 4, Magnum::Float>>
          matrix) {
    setUniform(uniformLocation("ciModelView"), matrix);
    return *this;
  }

  NBodyShader& setViewProjection(
      Corrade::Containers::ArrayView<
          const Magnum::Math::RectangularMatrix<4, 4, Magnum::Float>>
          matrix) {
    setUniform(uniformLocation("ciModelViewProjection"), matrix);
    return *this;
  }

  NBodyShader& bindTexture(Magnum::GL::Texture2D& tex) {
    tex.bind(0);
    setUniform(uniformLocation("star_tex"), 0);
    return *this;
  }
};

class NBodyApp : public Magnum::Platform::Application {
  // -- GUI --
  // Distribution choice
  enum {
    UI_DISTRIB_CYLINDER = 0,
    UI_DISTRIB_SPHERE = 1,
  };
  int32_t m_ui_distrib_id = UI_DISTRIB_CYLINDER;

  // Distribution parameters
  struct {
    float min_radius = 0;
    float max_radius = 25;
    float min_angle_pis = 0;
    float max_angle_pis = 2;
    float min_height = -50;
    float max_height = 50;
    float lg_speed = 0.4;
  } m_ui_distrib_cylinder_params;

  struct {
    float min_radius = 0;
    float max_radius = 25;
  } m_ui_distrib_sphere_params;

  int32_t m_ui_n_bodies = 1024;

  const int32_t m_num_updates_per_frame = 1;
  int32_t m_num_updates = 0;

  // Whether 'initialize' was clicked
  bool m_ui_initialize = false;

  // Whether the simulation is paused
  bool m_ui_paused = true;

  // Whether the user has requested a single step to be computed
  bool m_ui_step = false;

  // Force choice
  enum {
    UI_FORCE_GRAVITY = 0,
    UI_FORCE_LJ = 1,
    UI_FORCE_COULOMB = 2,
  };
  int32_t m_ui_force_id = UI_FORCE_GRAVITY;

  // Force parameters
  struct {
    float lg_G = -1.4;
    float lg_damping = -3;
  } m_ui_force_gravity_params;

  struct {
    float eps = 1;
    float lg_sigma = -5;
  } m_ui_force_lj_params;

  std::array<char, 256> m_ui_force_coulomb_file;

  // Integrator choice
  enum {
    UI_INTEGRATOR_EULER = 0,
    UI_INTEGRATOR_RK4 = 1,
  };
  int32_t m_ui_integrator_id = UI_INTEGRATOR_EULER;

  // -- PROGRAM VARIABLES --
  size_t m_n_bodies = m_ui_n_bodies;

  Magnum::Matrix4 m_view;
  Magnum::Matrix4 m_viewProjection;
  Corrade::Containers::Array<char> m_vboStorage;
  Magnum::GL::Mesh m_mesh;
  Magnum::GL::Texture2D m_star_tex;
  NBodyShader m_shader;
  Magnum::ImGuiIntegration::Context m_imgui{Magnum::NoCreate};

  // The simulation
  GravSim<num_t> m_sim;

 public:
  NBodyApp(const Arguments& arguments)
      : Magnum::Platform::
            Application{arguments,
                        Configuration{}
                            .setTitle("Codeplay NBody Demo")
                            .addWindowFlags(
                                Configuration::WindowFlag::Resizable),
                        GLConfiguration{}.setFlags(
                            GLConfiguration::Flag::QuietLog)},
        m_imgui{Magnum::Vector2{windowSize()} / dpiScaling(), windowSize(),
                framebufferSize()},
        m_sim(m_n_bodies, distrib_cylinder<num_t>{}) {
    const Corrade::Utility::Resource rs{"nbody-data"};
    Corrade::PluginManager::Manager<Magnum::Trade::AbstractImporter> manager;
    auto importer{manager.loadAndInstantiate("AnyImageImporter")};
    if (importer == nullptr || !importer->openData(rs.getRaw("star.png"))) {
      throw std::runtime_error{"Failed to load star.png"};
    }
    auto image{importer->image2D(0)};

    // Enable setting point size in shader
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Create star texture and bind it to GL slot 0
    m_star_tex.setWrapping(Magnum::GL::SamplerWrapping::ClampToEdge)
        .setMagnificationFilter(Magnum::GL::SamplerFilter::Linear)
        .setMinificationFilter(Magnum::GL::SamplerFilter::Linear)
        .setStorage(1, Magnum::GL::textureFormat(image->format()),
                    image->size())
        .setSubImage(0, {}, *image);

    m_ui_force_coulomb_file.fill(0);

    // Start with a perspective camera looking at the center
    using namespace Magnum::Math::Literals;
    m_view = Magnum::Matrix4::lookAt(Magnum::Vector3{200}, Magnum::Vector3{0},
                                     Magnum::Vector3{0, 0, 1})
                 .invertedRigid();
    m_viewProjection = Magnum::Matrix4::perspectiveProjection(
        60.0_degf, Magnum::Vector2{windowSize()}.aspectRatio(), 0.01f,
        50000.0f);

    init_gl_bufs();

    // Start the simulation with the application
    m_ui_initialize = true;
    m_ui_paused = false;
  }

  // Initializes the GL buffer for star position data with the current number of
  // bodies
  void init_gl_bufs() {
    const size_t arraySize{m_n_bodies * sizeof(sycl::vec<num_t, 3>)};
    m_vboStorage =
        Corrade::Containers::Array<char>(Corrade::ValueInit, 2 * arraySize);
    m_mesh = Magnum::GL::Mesh{Magnum::GL::MeshPrimitive::Points};
    m_mesh.setCount(m_n_bodies);
  }

  void tickEvent() override {
    // Initialize simulation if requested in UI
    if (m_ui_initialize) {
      m_n_bodies = m_ui_n_bodies;

      if (m_ui_force_id == UI_FORCE_COULOMB) {
        printf("Loading Coulomb data from %s\n",
               m_ui_force_coulomb_file.data());

        // First line is the particle count
        std::ifstream fin{m_ui_force_coulomb_file.data()};
        fin >> m_n_bodies;

        // Following lines are particle data
        std::vector<particle_data<num_t>> particles(m_n_bodies);
        for (auto& particle : particles) {
          num_t pos[3];
          fin >> particle.charge >> pos[0] >> pos[1] >> pos[2];
          particle.pos = {pos[0], pos[1], pos[2]};
        }

        m_sim = GravSim<num_t>(m_n_bodies, std::move(particles));
      } else if (m_ui_distrib_id == UI_DISTRIB_CYLINDER) {
        m_sim = GravSim<num_t>(
            m_n_bodies,
            distrib_cylinder<num_t>{
                {m_ui_distrib_cylinder_params.min_radius,
                 m_ui_distrib_cylinder_params.max_radius},
                {m_ui_distrib_cylinder_params.min_angle_pis * PI,
                 m_ui_distrib_cylinder_params.max_angle_pis * PI},
                {m_ui_distrib_cylinder_params.min_height,
                 m_ui_distrib_cylinder_params.max_height},
                sycl::pow(num_t(10), m_ui_distrib_cylinder_params.lg_speed)});
      } else if (m_ui_distrib_id == UI_DISTRIB_SPHERE) {
        m_sim = GravSim<num_t>(
            m_n_bodies,
            distrib_sphere<num_t>{{m_ui_distrib_sphere_params.min_radius,
                                   m_ui_distrib_sphere_params.max_radius}});
      }

      init_gl_bufs();

      m_ui_initialize = false;
    }

    if (!m_ui_paused || m_ui_step) {
      // Update force parameters
      switch (m_ui_force_id) {
        case UI_FORCE_GRAVITY: {
          m_sim.set_grav_G(
              sycl::pow(num_t(10), m_ui_force_gravity_params.lg_G));
          m_sim.set_grav_damping(
              sycl::pow(num_t(10), m_ui_force_gravity_params.lg_damping));
          m_sim.set_force_type(force_t::GRAVITY);
        } break;

        case UI_FORCE_LJ: {
          m_sim.set_lj_eps(m_ui_force_lj_params.eps);
          m_sim.set_lj_sigma(
              sycl::pow(num_t(10), m_ui_force_lj_params.lg_sigma));
          m_sim.set_force_type(force_t::LENNARD_JONES);
        } break;

        case UI_FORCE_COULOMB: {
          m_sim.set_force_type(force_t::COULOMB);
        } break;

        default:
          throw "unreachable";
      }

      // Update integration method
      switch (m_ui_integrator_id) {
        case UI_INTEGRATOR_EULER: {
          m_sim.set_integrator(integrator_t::EULER);
        } break;

        case UI_INTEGRATOR_RK4: {
          m_sim.set_integrator(integrator_t::RK4);
        } break;

        default:
          throw "unreachable";
      }

      // Run simulation frame
      if (m_ui_step) {
        m_sim.sync_queue();

        // Measure submission, execution and sync
        auto tstart = std::chrono::high_resolution_clock::now();
        m_sim.step();
        m_sim.sync_queue();
        auto tend = std::chrono::high_resolution_clock::now();

        // Convert to seconds
        auto diff = tend - tstart;
        auto sdiff = std::chrono::duration_cast<
                         std::chrono::duration<num_t, std::ratio<1, 1>>>(diff)
                         .count();

        std::cout << "Time taken for step: " << sdiff << "s" << std::endl;
      } else {
        for (int32_t step = 0; step < m_num_updates_per_frame; ++step) {
          m_sim.step();
        }
      }

      // Make sure not to step until clicked again
      m_ui_step = false;
    }

    m_num_updates++;
  }

  void drawEvent() override {
    Magnum::GL::defaultFramebuffer.clear(Magnum::GL::FramebufferClear::Depth)
        .clearColor(Magnum::Math::Color4{0.0f, 0.0f, 0.0f, 1.0f});

    // Disable depth to avoid black outlines
    Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::DepthTest);

    // Colors add to a bright white with additive blending
    Magnum::GL::Renderer::setBlendEquation(
        Magnum::GL::Renderer::BlendEquation::Add,
        Magnum::GL::Renderer::BlendEquation::Add);
    Magnum::GL::Renderer::setBlendFunction(
        Magnum::GL::Renderer::BlendFunction::SourceAlpha,
        Magnum::GL::Renderer::BlendFunction::OneMinusSourceAlpha);
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::Blending);

    // Update star buffer data with new positions
    const size_t arraySize{m_n_bodies * sizeof(sycl::vec<num_t, 3>)};
    m_sim.with_mapped(read_bufs_t<1>{},
                      [&](sycl::vec<num_t, 3> const* positions) {
                        std::copy_n(reinterpret_cast<const char*>(positions),
                                    arraySize, m_vboStorage.data());
                      });
    m_sim.with_mapped(read_bufs_t<0>{},
                      [&](sycl::vec<num_t, 3> const* velocities) {
                        std::copy_n(reinterpret_cast<const char*>(velocities),
                                    arraySize, m_vboStorage.data() + arraySize);
                      });

    Magnum::GL::Buffer vbo{m_vboStorage};
    m_mesh.addVertexBuffer(vbo, 0, NBodyShader::Position{})
        .addVertexBuffer(vbo, arraySize, NBodyShader::Velocity{});

    // Draw bodies
    m_shader.setView({&m_view, 1})
        .setViewProjection({&m_viewProjection, 1})
        .bindTexture(m_star_tex)
        .draw(m_mesh);

    // TODO: Port the Cinder arrow drawing to Magnum
    /*
    // Draw coordinate system arrows
    ci::gl::color(ci::Color(1.0f, 0.0f, 0.0f));
    ci::gl::drawVector(ci::vec3(90, 0, 0), ci::vec3(100, 0, 0), 2, 2);
    ci::gl::color(ci::Color(0.0f, 1.0f, 0.0f));
    ci::gl::drawVector(ci::vec3(0, 90, 0), ci::vec3(0, 100, 0), 2, 2);
    ci::gl::color(ci::Color(0.0f, 0.0f, 1.0f));
    ci::gl::drawVector(ci::vec3(0, 0, 90), ci::vec3(0, 0, 100), 2, 2);
    */

    // Draw the UI
    m_imgui.newFrame();

    /* Enable text input, if needed */
    if (ImGui::GetIO().WantTextInput && !isTextInputActive()) {
      startTextInput();
    } else if (!ImGui::GetIO().WantTextInput && isTextInputActive()) {
      stopTextInput();
    }

    ImGui::Begin("Simulation Settings");

    std::array<const char*, 3> forces = {
        {"Gravity", "Lennard-Jones", "Coulomb"}};
    ImGui::ListBox("Type of force", &m_ui_force_id, forces.data(),
                   forces.size(), forces.size());

    std::array<const char*, 2> integrators = {
        {"Euler [fast, inaccurate]", "RK4 [slow, accurate]"}};
    ImGui::ListBox("Integrator", &m_ui_integrator_id, integrators.data(),
                   integrators.size(), integrators.size());

    switch (m_ui_force_id) {
      case UI_FORCE_GRAVITY: {
        if (ImGui::TreeNode("Gravity settings")) {
          ImGui::SliderFloat("G constant [lg]", &m_ui_force_gravity_params.lg_G,
                             -8, 2);

          ImGui::SliderFloat("Damping factor [lg]",
                             &m_ui_force_gravity_params.lg_damping, -14, 0);

          ImGui::TreePop();
        }
      } break;

      case UI_FORCE_LJ: {
        if (ImGui::TreeNode("Lennard-Jones settings")) {
          ImGui::SliderFloat("Potential well depth", &m_ui_force_lj_params.eps,
                             0.1, 10);

          ImGui::SliderFloat("Zero potential radius [lg]",
                             &m_ui_force_lj_params.lg_sigma, -8, -2);

          ImGui::TreePop();
        }
      } break;

      case UI_FORCE_COULOMB: {
        if (ImGui::TreeNode("Coulomb settings")) {
          ImGui::Text(
              "Data format:\nLine 1: particle count (N)\nLines 2-(N+1): "
              "<charge> <x> <y> <z>");
          ImGui::InputText("Data input file", m_ui_force_coulomb_file.data(),
                           m_ui_force_coulomb_file.size());

          if (ImGui::Button("Initialize from file")) {
            m_ui_initialize = true;
          }

          ImGui::TreePop();
        }

      } break;

      default:
        throw std::runtime_error("unreachable");
    }

    if (ImGui::TreeNode("Initialization")) {
      std::array<const char*, 2> distribs = {{"Cylinder", "Sphere"}};
      ImGui::ListBox("Distribution", &m_ui_distrib_id, distribs.data(),
                     distribs.size(), distribs.size());

      ImGui::SliderInt("Number of bodies", &m_ui_n_bodies, 128, 16384);

      switch (m_ui_distrib_id) {
        case UI_DISTRIB_CYLINDER: {
          if (ImGui::TreeNode("Cylinder distribution settings")) {
            ImGui::DragFloatRange2(
                "Radius", &m_ui_distrib_cylinder_params.min_radius,
                &m_ui_distrib_cylinder_params.max_radius, 0.1f, 0.0f, 100.0f);

            ImGui::DragFloatRange2(
                "Angle [pi]", &m_ui_distrib_cylinder_params.min_angle_pis,
                &m_ui_distrib_cylinder_params.max_angle_pis, 0.01f, 0.0f, 2.0f);

            ImGui::DragFloatRange2("Height",
                                   &m_ui_distrib_cylinder_params.min_height,
                                   &m_ui_distrib_cylinder_params.max_height,
                                   0.1f, -100.0f, 100.0f);

            ImGui::SliderFloat("Speed [lg]",
                               &m_ui_distrib_cylinder_params.lg_speed, -3, 1);

            if (ImGui::Button("Initialize from distribution")) {
              m_ui_initialize = true;
            }

            ImGui::TreePop();
          }
        } break;

        case UI_DISTRIB_SPHERE: {
          if (ImGui::TreeNode("Sphere distribution settings")) {
            ImGui::DragFloatRange2(
                "Radius", &m_ui_distrib_sphere_params.min_radius,
                &m_ui_distrib_sphere_params.max_radius, 0.1f, 0.0f, 100.0f);

            if (ImGui::Button("Initialize from distribution")) {
              m_ui_initialize = true;
            }

            ImGui::TreePop();
          }
        } break;

        default:
          throw std::runtime_error("unreachable");
      }

      ImGui::TreePop();
    }

    if (m_ui_paused) {
      if (ImGui::Button("Start")) {
        m_ui_paused = false;
      }
      if (ImGui::Button("Step")) {
        m_ui_step = true;
      }
    } else {
      if (ImGui::Button("Pause")) {
        m_ui_paused = true;
      }
    }

    ImGui::End();

    m_imgui.updateApplicationCursor(*this);

    Magnum::GL::Renderer::setBlendEquation(
        Magnum::GL::Renderer::BlendEquation::Add,
        Magnum::GL::Renderer::BlendEquation::Add);
    Magnum::GL::Renderer::setBlendFunction(
        Magnum::GL::Renderer::BlendFunction::SourceAlpha,
        Magnum::GL::Renderer::BlendFunction::OneMinusSourceAlpha);
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::Blending);
    Magnum::GL::Renderer::enable(Magnum::GL::Renderer::Feature::ScissorTest);
    Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::FaceCulling);
    Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::DepthTest);
    m_imgui.drawFrame();

    redraw();
    swapBuffers();
  }

  void viewportEvent(ViewportEvent& event) override {
    Magnum::GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});

    m_imgui.relayout(Magnum::Vector2{event.windowSize()} / event.dpiScaling(),
                     event.windowSize(), event.framebufferSize());
  }

  void keyPressEvent(KeyEvent& event) override {
    m_imgui.handleKeyPressEvent(event);
  }
  void keyReleaseEvent(KeyEvent& event) override {
    m_imgui.handleKeyReleaseEvent(event);
  }

  void mousePressEvent(MouseEvent& event) override {
    m_imgui.handleMousePressEvent(event);
  }
  void mouseReleaseEvent(MouseEvent& event) override {
    m_imgui.handleMouseReleaseEvent(event);
  }
  void mouseMoveEvent(MouseMoveEvent& event) override {
    m_imgui.handleMouseMoveEvent(event);
  }
  void mouseScrollEvent(MouseScrollEvent& event) override {
    if (m_imgui.handleMouseScrollEvent(event)) {
      /* Prevent scrolling the page */
      event.setAccepted();
      return;
    }
  }

  void textInputEvent(TextInputEvent& event) override {
    m_imgui.handleTextInputEvent(event);
  }
};

MAGNUM_APPLICATION_MAIN(NBodyApp)
