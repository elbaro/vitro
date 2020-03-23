#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <experimental/filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

/// hold python objects
class Line {
public:
  // Line& width(double width) {
  //   width_ = width;
  //   return *this;
  // }
  // Line& alpha(double alpha) {
  //   alpha_ = alpha;
  //   return *this;
  // }

  // private:
  std::string name{};
  double width = 1.0;
  double alpha = 0.8;
  std::optional<std::string> color;
  std::string drawstyle = "default";
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
  // Scatter& alpha(double alpha) {
  //   alpha_ = alpha;
  //   return *this;
  // }

  // private:
  std::string name;
  std::string marker_type = "o"; // temporarily we use matplotlib style
  double marker_size = 10;
  std::optional<std::string> marker_edge_color; // accepts "none"
  std::optional<std::string> marker_face_color; // accepts "none"
  std::optional<std::string> color;
  double width = 1.0;
  double alpha = 0.8;
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
  // Area& alpha(double alpha) {
  //   alpha_ = alpha;
  //   return *this;
  // }

  // private:
  std::string name;
  std::string color = "r";
  double alpha = 0.8;

  std::vector<int64_t> xs; // TODO : generic
  std::vector<double> y1s;
  std::vector<double> y2s;
};

// class Bar {
// public:
//   std::string name;
//   double alpha = 0.4;

//   std::vector<double> xs;
//   std::vector<double> ys;
// }

class Histogram {
public:
  std::vector<std::string> names;
  double alpha = 0.3;
  bool stacked = false;
  int cumulative = 0;              // 1, 0, -1
  std::string type = "stepfilled"; //  or "step"
  bool normalize_area = false;

  std::vector<std::vector<double>> xs_list;
  std::optional<std::vector<std::vector<double>>> weights_list;
  int num_bins = 0;
  bool bin_log_scale = false;
};

class Text {
public:
  double alpha = 0.8;
  double fontsize = 9.0;
  std::string text;
  double x; // 0~1
  double y; // 0~1
  std::string horizontalalignment = "right";
  std::string verticalalignment = "bottom";
  std::string transform = "axes"; // data, axes, figure

  // bbox
  double bbox_alpha = 0.5;
  std::string bbox_color = "wheat";
  std::string bbox_style = "round";
};

class Axes {
public:
  std::pair<int64_t, int64_t> xrange();
  /* ---------------------------------- plot ---------------------------------- */
  Line& line(const std::string& name, const std::vector<int64_t>& xs, const std::vector<double>& ys);
  Scatter& scatter(const std::string& name, const std::vector<int64_t>& xs, const std::vector<double>& ys);
  Area& area(const std::string& name, const std::vector<int64_t>& xs, const std::vector<double>& y1s,
             const std::vector<double>& y2s);
  Histogram& histogram(const std::vector<std::string>& names, int num_bins,
                       const std::vector<std::vector<double>>& xs_list);
  Text& text(const std::string& text, double x, double y);

  std::string title{};
  std::string legend = "best"; // use "" to disable
  std::string xlabel{};
  std::string ylabel{};
  std::optional<int64_t> xlim_left;
  std::optional<int64_t> xlim_right;
  std::string xscale{}; // "log", "logit", ..
  std::string yscale{};

  bool grid = false;
  std::string grid_tick = "both"; // "major", "minor"
  std::string grid_axis = "both"; // "x", "y"

  std::string xtick_major = "auto"; // "auto" / "max_n"+xtick_n / "multiple"+xtick_multiple
  int xtick_major_n = 10;
  double xtick_major_multiple = 1.0;

  bool is_x_nanotimestamps = false; // if true, x values are interprested as unix epochs in nanoseconds
  std::vector<Line> lines;
  std::vector<Scatter> scatters;
  std::vector<Area> areas;
  std::vector<Histogram> histograms;
  std::vector<Text> texts;
  bool is_twin = false;
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

  Axes& twinx(int r, int c) {
    if (r < 1 || c < 1 || r > nrow || c > ncol) {
      throw std::invalid_argument("axes out of range");
    }
    auto pair = std::make_pair(r, c);
    if (twinx_idx_.find(pair) == twinx_idx_.end()) {
      axes_.emplace_back();
      axes_.back().is_twin = true;
      twinx_idx_[pair] = axes_.size() - 1;
    }
    return axes_[twinx_idx_[pair]];
  }
  Axes& twiny(int r, int c) {
    if (r < 1 || c < 1 || r > nrow || c > ncol) {
      throw std::invalid_argument("axes out of range");
    }
    auto pair = std::make_pair(r, c);
    if (twiny_idx_.find(pair) == twiny_idx_.end()) {
      axes_.emplace_back();
      axes_.back().is_twin = true;
      twiny_idx_[pair] = axes_.size() - 1;
    }
    return axes_[twiny_idx_[pair]];
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
  std::optional<std::vector<double>> width_ratios;
  std::optional<std::vector<double>> height_ratios;

  std::vector<Axes> axes_; // grid + extra twinx, twiny
  std::map<std::pair<int, int>, int> twinx_idx_;
  std::map<std::pair<int, int>, int> twiny_idx_;
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
