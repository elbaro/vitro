#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <experimental/filesystem>
#include <string>
#include <vector>

/// hold python objects
class Line {
public:
  // Line& width(double width) {
  //   width_ = width;
  //   return *this;
  // }
  // Line& opacity(double opacity) {
  //   opacity_ = opacity;
  //   return *this;
  // }

  // private:
  std::string name{};
  double width_ = 1.0;
  double opacity_ = 0.0;
  std::vector<int64_t> xs{}; // TODO : generic
  std::vector<double> ys{};
};

class Scatter {
public:
  // Scatter& marker_type(char marker) {
  //   marker_type_ = marker;
  //   return *this;
  // }
  // Scatter& marker_size(double size) {
  //   marker_size_ = size;
  //   return *this;
  // }
  // Scatter& opacity(double opacity) {
  //   opacity_ = opacity;
  //   return *this;
  // }

  // private:
  std::string name;
  std::string marker_type = "o"; // temporarily we use matplotlib style
  double marker_size = 10;
  double opacity = 0.0;
  std::vector<int64_t> xs; // TODO : generic
  std::vector<double> ys;
};

class Area {
public:
  // Area& line_color(char color) {
  //   line_color_ = color;
  //   return *this;
  // }
  // Area& fill_color(char color) {
  //   fill_color_ = color;
  //   return *this;
  // }
  // Area& opacity(double opacity) {
  //   opacity_ = opacity;
  //   return *this;
  // }

  // private:
  std::string name;
  char line_color = 'b';
  char fill_color = 'r';
  double opacity = 0.0;
  std::vector<int64_t> xs; // TODO : generic
  std::vector<double> y1s;
  std::vector<double> y2s;
};

class Axes {
public:
  /* ---------------------------------- plot ---------------------------------- */
  Line& line(const std::string& name, const std::vector<int64_t>& xs, const std::vector<double>& ys);
  Scatter& scatter(const std::string& name, const std::vector<int64_t>& xs, const std::vector<double>& ys);
  Area& area(const std::string& name, const std::vector<int64_t>& xs, const std::vector<double>& y1s,
             const std::vector<double>& y2s);

  std::string title{};
  std::string xlabel;
  std::string ylabel;
  bool is_x_nanotimestamps = false; // if true, x values are interprested as unix epochs in nanoseconds
  std::vector<Line> lines;
  std::vector<Scatter> scatters;
  std::vector<Area> areas;
};

class Figure {
public:
  Figure() : Figure(1, 1){};
  Figure(int r, int c) : nrow(r), ncol(c), axes_(r * c) {}

  const Axes& readonly_axes(int r, int c) const {
    if (r < 1 || c < 1 || r > nrow || c > ncol) {
      throw std::invalid_argument("axes out of range");
    }
    return axes_[(r - 1) * ncol + c - 1];
  }
  Axes& axes(int r, int c) {
    if (r < 1 || c < 1 || r > nrow || c > ncol) {
      throw std::invalid_argument("axes out of range");
    }
    return axes_[(r - 1) * ncol + c - 1];
  }

  /* -------------------------------- configure ------------------------------- */
  // Figure& title(const std::string& title) {}
  // Figure& size(int width, int height) {
  //   width_ = width;
  //   height_ = height;
  //   return *this;
  // }

  const int nrow;
  const int ncol;
  double width = 800;
  double height = 600;
  std::string title;

private:
  std::vector<Axes> axes_;
};

class Matplot {
public:
  Matplot(const Figure& fig);
  ~Matplot() { Py_DECREF(pyfig); }

  /* --------------------------------- export --------------------------------- */
  void save(const std::experimental::filesystem::path& path);

private:
  PyObject* pyfig;
};