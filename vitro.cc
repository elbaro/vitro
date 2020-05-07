#include "vitro.h"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <python3.8/object.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include <numpy/arrayobject.h>

void* _import_numpy() {
  import_array();
  return nullptr;
}

PyObject* new_none() { Py_RETURN_NONE; }

void check_pyerr() {
  if (PyErr_Occurred()) {
    PyErr_Print();
    throw std::runtime_error("error from python");
  }
}

void make_sure_initialized() {
  static std::atomic<bool> should_be_initialized(true);
  if (should_be_initialized.exchange(false, std::memory_order_seq_cst)) {
    Py_Initialize();
    _import_numpy();
  }
}

template <typename T> struct select_npy_type { const static NPY_TYPES type = NPY_NOTYPE; }; // default
template <> struct select_npy_type<double> { const static NPY_TYPES type = NPY_DOUBLE; };
template <> struct select_npy_type<int64_t> { const static NPY_TYPES type = NPY_INT64; };
template <> struct select_npy_type<PyObject*> { const static NPY_TYPES type = NPY_OBJECT; };

template <typename V> PyObject* vector_to_1darray(const std::vector<V>& vec) {
  Py_intptr_t dims[] = {static_cast<Py_intptr_t>(vec.size())};
  auto arr = PyArray_SimpleNewFromData(1, dims, select_npy_type<V>::type,
                                       const_cast<void*>(reinterpret_cast<const void*>(vec.data())));
  if (arr == nullptr) {
    throw std::runtime_error("cannot convert std::vector to np.ndarray");
  }
  return arr;
}

static_assert(sizeof(Py_intptr_t) == sizeof(int*));

template <typename V> PyObject* vector_to_ndarray(const std::vector<V>& vec, const std::vector<int> dims) {
  const Py_intptr_t* pydims = reinterpret_cast<const Py_intptr_t*>(dims.data());

  auto arr = PyArray_SimpleNewFromData(dims.size(), pydims, select_npy_type<V>::type,
                                       const_cast<void*>(reinterpret_cast<const void*>(vec.data())));
  if (arr == nullptr) {
    throw std::runtime_error("cannot convert std::vector to np.ndarray");
  }
  return arr;
}

PyObject* slow_int64_to_datetime(int64_t ts) {
  PyObject* datetime_module = PyImport_ImportModule("datetime");
  if (!datetime_module) {
    throw std::runtime_error("Error loading module datetime");
  }
  PyObject* datetime = PyObject_GetAttrString(datetime_module, "datetime");
  PyObject* utcfromtimestamp = PyObject_GetAttrString(datetime, "utcfromtimestamp");
  PyObject* out = PyObject_CallFunction(utcfromtimestamp, "d", static_cast<double>(ts) * 1e-9);
  Py_DecRef(utcfromtimestamp);
  Py_DecRef(datetime);
  Py_DecRef(datetime_module);
  return out;
}

class Timestamps {
public:
  Timestamps(const std::vector<int64_t>& epochs) {
    make_sure_initialized();

    PyObject* datetime_module = PyImport_ImportModule("datetime");
    if (!datetime_module) {
      throw std::runtime_error("Error loading module datetime");
    }
    PyObject* datetime = PyObject_GetAttrString(datetime_module, "datetime");
    PyObject* utcfromtimestamp = PyObject_GetAttrString(datetime, "utcfromtimestamp");

    datetimes.reserve(epochs.size());
    for (auto epoch : epochs) {
      PyObject* result;
      result = PyObject_CallFunction(utcfromtimestamp, "d", static_cast<double>(epoch) * 1e-9);
      datetimes.push_back(result);
    }

    Py_DecRef(utcfromtimestamp);
    Py_DecRef(datetime);
    Py_DecRef(datetime_module);
  }

  ~Timestamps() {
    for (PyObject* datetime : datetimes) {
      Py_DecRef(datetime);
    }
  }

  std::vector<PyObject*> datetimes;
};

Line& Axes::line(const std::string& name, const std::vector<int64_t>& xs, const std::vector<double>& ys) {
  lines.emplace_back();
  lines.back().name = name;
  lines.back().xs = xs;
  lines.back().ys = ys;
  return lines.back();
}

Scatter& Axes::scatter(const std::string& name, const std::vector<int64_t>& xs, const std::vector<double>& ys) {
  scatters.emplace_back();
  scatters.back().name = name;
  scatters.back().xs = xs;
  scatters.back().ys = ys;
  return scatters.back();
}

Area& Axes::area(const std::string& name, const std::vector<int64_t>& xs, const std::vector<double>& y1s,
                 const std::vector<double>& y2s) {
  areas.emplace_back();
  areas.back().name = name;
  areas.back().xs = xs;
  areas.back().y1s = y1s;
  areas.back().y2s = y2s;
  return areas.back();
}

Histogram& Axes::histogram(const std::vector<std::string>& names, int num_bins,
                           const std::vector<std::vector<double>>& xs_list) {
  histograms.emplace_back();
  histograms.back().names = names;
  histograms.back().num_bins = num_bins;
  histograms.back().xs_list = xs_list;
  return histograms.back();
}

Text& Axes::text(const std::string& text, double x, double y) {
  texts.emplace_back();
  texts.back().text = text;
  texts.back().x = x;
  texts.back().y = y;
  return texts.back();
}

std::pair<int64_t, int64_t> Axes::xrange() {
  int64_t mn = std::numeric_limits<int64_t>::max();
  int64_t mx = std::numeric_limits<int64_t>::min();
  for (const auto& line : lines) {
    if (line.xs.size() > 0) {
      auto v1 = *std::min_element(line.xs.begin(), line.xs.end());
      auto v2 = *std::max_element(line.xs.begin(), line.xs.end());
      if (v1 < mn) {
        mn = v1;
      }
      if (v2 > mx) {
        mx = v2;
      }
    }
  }
  for (const auto& scatter : scatters) {
    if (scatter.xs.size() > 0) {
      auto v1 = *std::min_element(scatter.xs.begin(), scatter.xs.end());
      auto v2 = *std::max_element(scatter.xs.begin(), scatter.xs.end());
      if (v1 < mn) {
        mn = v1;
      }
      if (v2 > mx) {
        mx = v2;
      }
    }
  }
  for (const auto& area : areas) {
    if (area.xs.size() > 0) {
      auto v1 = *std::min_element(area.xs.begin(), area.xs.end());
      auto v2 = *std::max_element(area.xs.begin(), area.xs.end());
      if (v1 < mn) {
        mn = v1;
      }
      if (v2 > mx) {
        mx = v2;
      }
    }
  }
  // for (const auto& h : histograms) {
  //   if (area.xs.size() > 0) {
  //     auto v1 = *std::min_element(area.xs.begin(), area.xs.end());
  //     auto v2 = *std::max_element(area.xs.begin(), area.xs.end());
  //     if (v1 < mn) {
  //       mn = v1;
  //     }
  //     if (v2 > mx) {
  //       mx = v2;
  //     }
  //   }
  // }
  return std::make_pair(mn, mx);
}

Matplot::Matplot(const Figure& fig) {
  make_sure_initialized();

  PyObject* pyplot = PyImport_ImportModule("matplotlib.pyplot");
  if (!pyplot) {
    throw std::runtime_error("Error importing matplotlib.pyplot");
  }

  PyObject* numpy = PyImport_ImportModule("numpy");
  if (!numpy) {
    throw std::runtime_error("Error importing module numpy");
  }

  PyObject* py_matplotlib_ticker = PyImport_ImportModule("matplotlib.ticker");
  if (!py_matplotlib_ticker) {
    throw std::runtime_error("Error importing module numpy");
  }

  // default dpi = 100 = pixels per inch
  // width (px) => width/dpi (inches)
  pyfig = PyObject_CallMethod(pyplot, "figure", "O(dd)", new_none(), fig.width / 100.0, fig.height / 100.0);
  // pyfig = PyObject_CallMethod(pyplot, "figure", nullptr);
  if (pyfig == nullptr) {
    throw std::runtime_error("matplotlib.pyplot.figure() failed");
  }

  if (fig.title.size() > 0) {
    PyObject_CallMethod(pyfig, "suptitle", "s", fig.title.c_str());
  }

  // gridspec_kw
  PyObject* gridspec_kw = nullptr;
  if (fig.width_ratios || fig.height_ratios) {
    gridspec_kw = PyDict_New();
    if (fig.width_ratios) {
      auto list = PyList_New(fig.width_ratios->size());
      Py_ssize_t idx = 0;
      for (auto& ratio : *fig.width_ratios) {
        PyList_SetItem(list, idx++, PyFloat_FromDouble(ratio));
      }
      PyDict_SetItemString(gridspec_kw, "width_ratios", list);
    }
    if (fig.height_ratios) {
      auto list = PyList_New(fig.height_ratios->size());
      Py_ssize_t idx = 0;
      for (auto& ratio : *fig.height_ratios) {
        PyList_SetItem(list, idx++, PyFloat_FromDouble(ratio));
      }
      PyDict_SetItemString(gridspec_kw, "height_ratios", list);
    }
    Py_IncRef(gridspec_kw);
  }

  // r,c,sharex,sharey,squeeze,subplot_kw,gridspec_kw
  PyObject* result = PyObject_CallMethod(pyfig, "subplots", "iiOOOOO", fig.nrow, fig.ncol, Py_False, Py_False, Py_False,
                                         Py_None, (gridspec_kw == nullptr) ? new_none() : gridspec_kw);
  if (result == nullptr) {
    throw std::runtime_error("Figure.subplots() failed");
  }

  std::vector<PyObject*> pyaxes(fig.axes_.size());
  for (Py_intptr_t i = 0; i < fig.nrow; i++) {
    for (Py_intptr_t j = 0; j < fig.ncol; j++) {
      int idx = i * fig.ncol + j;
      auto* ax = pyaxes[idx] =
          *reinterpret_cast<PyObject**>(PyArray_GETPTR2(reinterpret_cast<PyArrayObject*>(result), i, j));
      Py_IncRef(ax);

      auto pair = std::make_pair(static_cast<int>(i + 1), static_cast<int>(j + 1));
      if (fig.twinx_idx_.find(pair) != fig.twinx_idx_.end()) {
        auto* twin_ax = pyaxes[fig.twinx_idx_.at(pair)] = PyObject_CallMethod(ax, "twinx", nullptr);
        auto* ax_get_lines = PyObject_GetAttrString(ax, "_get_lines");
        auto* ax_prop_cycler = PyObject_GetAttrString(ax_get_lines, "prop_cycler");
        auto* twin_get_lines = PyObject_GetAttrString(twin_ax, "_get_lines");
        PyObject_SetAttrString(twin_get_lines, "prop_cycler", ax_prop_cycler);
        Py_DecRef(ax_prop_cycler);
        Py_DecRef(ax_get_lines);
        Py_DecRef(twin_get_lines);
      }
      if (fig.twiny_idx_.find(pair) != fig.twiny_idx_.end()) {
        auto* twin_ax = pyaxes[fig.twiny_idx_.at(pair)] = PyObject_CallMethod(ax, "twiny", nullptr);
        auto* ax_get_lines = PyObject_GetAttrString(ax, "_get_lines");
        auto* ax_prop_cycler = PyObject_GetAttrString(ax_get_lines, "prop_cycler");
        auto* twin_get_lines = PyObject_GetAttrString(twin_ax, "_get_lines");
        PyObject_SetAttrString(twin_get_lines, "prop_cycler", ax_prop_cycler);
        Py_DecRef(ax_prop_cycler);
        Py_DecRef(ax_get_lines);
        Py_DecRef(twin_get_lines);
      }
    }
  }
  Py_DecRef(result);

  for (size_t ax_i = 0; ax_i < fig.axes_.size(); ax_i++) {
    const auto& ax = fig.axes_[ax_i];
    auto* pyax = pyaxes[ax_i];
    if (ax.title.size() > 0) {
      PyObject_CallMethod(pyax, "set_title", "s", ax.title.c_str());
      check_pyerr();
    }
    if (ax.xlabel.size() > 0) {
      PyObject_CallMethod(pyax, "set_xlabel", "s", ax.xlabel.c_str());
      check_pyerr();
    }
    if (ax.ylabel.size() > 0) {
      PyObject_CallMethod(pyax, "set_ylabel", "s", ax.ylabel.c_str());
      check_pyerr();
    }
    if (ax.xscale.size() > 0) {
      PyObject_CallMethod(pyax, "set_xscale", "s", ax.xscale.c_str());
      check_pyerr();
    }
    if (ax.yscale.size() > 0) {
      PyObject_CallMethod(pyax, "set_yscale", "s", ax.yscale.c_str());
      check_pyerr();
    }
    if (ax.grid) {
      PyObject_CallMethod(pyax, "grid", "Oss", Py_True, ax.grid_tick.c_str(), ax.grid_axis.c_str());
      check_pyerr();
    }

    // TODO: new struct for xtick_major, xtick_minor, ytick_major, ytick_minor
    if (ax.xtick_major != "auto") {
      PyObject* locator;
      if (ax.xtick_major == "") {
        locator = PyObject_CallMethod(py_matplotlib_ticker, "NullLocator", nullptr);
      } else if (ax.xtick_major == "max_n") {
        locator = PyObject_CallMethod(py_matplotlib_ticker, "MaxNLocator", "i", ax.xtick_major_n);
      } else if (ax.xtick_major == "multiple") {
        locator = PyObject_CallMethod(py_matplotlib_ticker, "MultipleLocator", "d", ax.xtick_major_multiple);
      } else {
        throw std::runtime_error("cannot create matplotlib xtick locator");
      }
      check_pyerr();
      auto* xaxis = PyObject_GetAttrString(pyax, "xaxis");
      PyObject_CallMethod(xaxis, "set_major_locator", "O", locator);
      Py_DecRef(xaxis);
      Py_DecRef(locator);
    }
    if (ax.xtick_minor != "") {
      PyObject* locator;
      if (ax.xtick_minor == "auto") {
        locator = PyObject_CallMethod(py_matplotlib_ticker, "AutoMinorLocator", nullptr);
      } else if (ax.xtick_minor == "max_n") {
        locator = PyObject_CallMethod(py_matplotlib_ticker, "MaxNLocator", "i", ax.xtick_minor_n);
      } else if (ax.xtick_minor == "multiple") {
        locator = PyObject_CallMethod(py_matplotlib_ticker, "MultipleLocator", "d", ax.xtick_minor_multiple);
      } else {
        throw std::runtime_error("cannot create matplotlib xtick locator");
      }
      check_pyerr();
      auto* xaxis = PyObject_GetAttrString(pyax, "xaxis");
      PyObject_CallMethod(xaxis, "set_minor_locator", "O", locator);
      Py_DecRef(xaxis);
      Py_DecRef(locator);
    }
    if (ax.ytick_major != "auto") {
      PyObject* locator;
      if (ax.ytick_major == "") {
        locator = PyObject_CallMethod(py_matplotlib_ticker, "NullLocator", nullptr);
      } else if (ax.ytick_major == "max_n") {
        locator = PyObject_CallMethod(py_matplotlib_ticker, "MaxNLocator", "i", ax.ytick_major_n);
      } else if (ax.ytick_major == "multiple") {
        locator = PyObject_CallMethod(py_matplotlib_ticker, "MultipleLocator", "d", ax.ytick_major_multiple);
      } else {
        throw std::runtime_error("cannot create matplotlib ytick locator");
      }
      check_pyerr();
      auto* yaxis = PyObject_GetAttrString(pyax, "yaxis");
      PyObject_CallMethod(yaxis, "set_major_locator", "O", locator);
      Py_DecRef(yaxis);
      Py_DecRef(locator);
    }
    if (ax.ytick_minor != "") {
      PyObject* locator;
      if (ax.ytick_minor == "auto") {
        locator = PyObject_CallMethod(py_matplotlib_ticker, "AutoMinorLocator", nullptr);
      } else if (ax.ytick_minor == "max_n") {
        locator = PyObject_CallMethod(py_matplotlib_ticker, "MaxNLocator", "i", ax.ytick_minor_n);
      } else if (ax.ytick_minor == "multiple") {
        locator = PyObject_CallMethod(py_matplotlib_ticker, "MultipleLocator", "d", ax.ytick_minor_multiple);
      } else {
        throw std::runtime_error("cannot create matplotlib ytick locator");
      }
      check_pyerr();
      auto* yaxis = PyObject_GetAttrString(pyax, "yaxis");
      PyObject_CallMethod(yaxis, "set_minor_locator", "O", locator);
      Py_DecRef(yaxis);
      Py_DecRef(locator);
    }

    for (const auto& line : ax.lines) {
      PyObject* x;
      std::unique_ptr<Timestamps> ts;
      if (ax.is_x_nanotimestamps) {
        ts = std::make_unique<Timestamps>(line.xs);
        x = vector_to_1darray(ts->datetimes);
      } else {
        x = vector_to_1darray(line.xs);
      }
      auto y = vector_to_1darray(line.ys);

      auto plot = PyObject_GetAttrString(pyax, "plot");
      auto args = Py_BuildValue("(OO)", x, y);
      auto kwargs = Py_BuildValue("{s:s,s:d,s:s,s:d,s:O}", "label", line.name.c_str(), "alpha", line.alpha, "drawstyle",
                                  line.drawstyle.c_str(), "linewidth", line.width, "color",
                                  line.color ? PyUnicode_FromString(line.color->c_str()) : Py_None);
      auto pyline = PyObject_Call(plot, args, kwargs);
      if (pyline == nullptr) {
        throw std::runtime_error("cannot draw a line");
      }

      Py_DecRef(x);
      Py_DecRef(y);
      Py_DecRef(args);
      Py_DecRef(kwargs);
      Py_DecRef(plot);
      Py_DecRef(pyline);
    }
    for (const auto& scatter : ax.scatters) {
      PyObject* x;
      std::unique_ptr<Timestamps> ts;
      if (ax.is_x_nanotimestamps) {
        ts = std::make_unique<Timestamps>(scatter.xs);
        x = vector_to_1darray(ts->datetimes);
      } else {
        x = vector_to_1darray(scatter.xs);
      }
      auto y = vector_to_1darray(scatter.ys);

      auto plot = PyObject_GetAttrString(pyax, "scatter");
      auto args = Py_BuildValue("(OOd)", x, y, scatter.marker_size);
      auto color_str = scatter.color ? PyUnicode_FromString(scatter.color->c_str()) : Py_None;
      auto kwargs =
          Py_BuildValue("{s:s, s:d, s:s, s:d, s:O}", "label", scatter.name.c_str(), "alpha", scatter.alpha, "marker",
                        scatter.marker_type.c_str(), "linewidths", scatter.width, "color", color_str);
      if (scatter.marker_edge_color) {
        auto s = PyUnicode_FromString(scatter.marker_edge_color->c_str());
        PyDict_SetItemString(kwargs, "edgecolors", s);
        Py_DecRef(s);
      }
      if (scatter.marker_face_color) {
        auto s = PyUnicode_FromString(scatter.marker_face_color->c_str());
        PyDict_SetItemString(kwargs, "facecolors", s);
        Py_DecRef(s);
      }
      if (color_str != Py_None) {
        Py_DecRef(color_str);
      }
      auto pyscatter = PyObject_Call(plot, args, kwargs);
      check_pyerr();
      if (pyscatter == nullptr) {
        throw std::runtime_error("cannot draw a scatter");
      }

      Py_DecRef(x);
      Py_DecRef(y);
      Py_DecRef(args);
      Py_DecRef(kwargs);
      Py_DecRef(plot);
      Py_DecRef(pyscatter);
    }
    for (const auto& area : ax.areas) {
      PyObject* x;
      std::unique_ptr<Timestamps> ts;
      if (ax.is_x_nanotimestamps) {
        ts = std::make_unique<Timestamps>(area.xs);
        x = vector_to_1darray(ts->datetimes);
      } else {
        x = vector_to_1darray(area.xs);
      }
      auto y1 = vector_to_1darray(area.y1s);
      auto y2 = vector_to_1darray(area.y2s);

      auto plot = PyObject_GetAttrString(pyax, "fill_between");
      auto args = Py_BuildValue("(OOO)", x, y1, y2);
      auto kwargs = Py_BuildValue("{s:s, s:d, s:s}", "label", area.name.c_str(), "alpha", area.alpha, "facecolor",
                                  area.color.c_str());
      auto pyarea = PyObject_Call(plot, args, kwargs);
      check_pyerr();
      if (pyarea == nullptr) {
        throw std::runtime_error("cannot draw a fill_between");
      }
      Py_DecRef(x);
      Py_DecRef(y1);
      Py_DecRef(y2);
      Py_DecRef(args);
      Py_DecRef(kwargs);
      Py_DecRef(plot);
      Py_DecRef(pyarea);
    }
    for (const auto& histogram : ax.histograms) {
      PyObject* x = PyList_New(histogram.xs_list.size());
      {
        Py_ssize_t idx = 0;
        for (auto& xs : histogram.xs_list) {
          PyList_SetItem(x, idx++, vector_to_1darray(xs));
        }
      }

      PyObject* y = nullptr;
      if (histogram.weights_list) {
        y = PyList_New(histogram.weights_list->size());
        Py_ssize_t idx = 0;
        for (auto& ys : *histogram.weights_list) {
          PyList_SetItem(y, idx++, vector_to_1darray(ys));
        }
      }

      PyObject* labels = PyList_New(histogram.names.size());
      {
        Py_ssize_t idx = 0;
        for (auto& name : histogram.names) {
          PyList_SetItem(labels, idx++, PyUnicode_FromString(name.c_str()));
        }
      }

      PyObject* bins;
      if (histogram.bin_log_scale) {
        double min = std::numeric_limits<double>::max();
        double max = std::numeric_limits<double>::min();
        for (auto& xs : histogram.xs_list) {
          if (xs.empty()) {
            continue;
          }
          const auto [mn, mx] = std::minmax_element(xs.begin(), xs.end());
          if (*mn < min) {
            min = *mn;
          }
          if (*mx > max) {
            max = *mx;
          }
        }

        if (min <= 0) {
          throw std::invalid_argument("when drawing histogram in log_scale, input number should be > 0");
        }

        bins = PyObject_CallMethod(numpy, "geomspace", "(ddi)", min, max, histogram.num_bins);
        check_pyerr();
        if (bins == nullptr) {
          throw std::runtime_error("cannot call np.geomspace()");
        }
      } else {
        bins = PyLong_FromLong(histogram.num_bins);
      }

      auto plot = PyObject_GetAttrString(pyax, "hist");
      auto args = Py_BuildValue("(O)", x);
      auto kwargs = Py_BuildValue("{s:O,s:O,s:d,s:O,s:i,s:s,s:O,s:O}",                      //
                                  "weights", (y == nullptr) ? Py_None : y,                  //
                                  "label", labels,                                          //
                                  "alpha", histogram.alpha,                                 //
                                  "bins", bins,                                             //
                                  "cumulative", histogram.cumulative,                       //
                                  "histtype", histogram.type.c_str(),                       //
                                  "density", histogram.normalize_area ? Py_True : Py_False, //
                                  "stacked", histogram.stacked ? Py_True : Py_False);
      auto pyhist = PyObject_Call(plot, args, kwargs);
      check_pyerr();
      if (pyhist == nullptr) {
        throw std::runtime_error("cannot draw a histogram");
      }

      Py_DecRef(x);
      Py_DecRef(y);
      Py_DecRef(labels);
      Py_DecRef(bins);
      Py_DecRef(args);
      Py_DecRef(kwargs);
      Py_DecRef(plot);
      Py_DecRef(pyhist);
    }
    for (const auto& text : ax.texts) {
      auto props = Py_BuildValue("{s:s, s:s, s:d}", "boxstyle", text.bbox_style.c_str(), "facecolor",
                                 text.bbox_color.c_str(), "alpha", text.bbox_alpha);
      PyObject* transform;
      if (text.transform == "axes") {
        transform = PyObject_GetAttrString(pyax, "transAxes");
      } else {
        throw std::runtime_error("unknown text coordinate transform: " + text.transform);
      }
      auto plot = PyObject_GetAttrString(pyax, "text");
      auto args = Py_BuildValue("(dds)", text.x, text.y, text.text.c_str());
      auto kwargs =
          Py_BuildValue("{s:d,s:d,s:O,s:O,s:s,s:s}", "fontsize", text.fontsize, "alpha", text.alpha, "transform",
                        transform, "bbox", props, "verticalalignment", text.verticalalignment.c_str(),
                        "horizontalalignment", text.horizontalalignment.c_str());

      auto pytext = PyObject_Call(plot, args, kwargs);
      check_pyerr();
      if (pytext == nullptr) {
        throw std::runtime_error("cannot draw a text");
      }
      Py_DecRef(transform);
      Py_DecRef(props);
      Py_DecRef(args);
      Py_DecRef(kwargs);
      Py_DecRef(plot);
      Py_DecRef(pytext);
    }
    if (ax.lines.size() + ax.scatters.size() + ax.areas.size() + ax.histograms.size() > 0) {
      if (ax.legend.size() > 0) {
        auto legend = PyObject_GetAttrString(pyax, "legend");
        auto args = PyTuple_New(0);
        auto kwargs = Py_BuildValue("{s:s}", "loc", ax.legend.c_str());
        PyObject_Call(legend, args, kwargs);
        Py_DecRef(args);
        Py_DecRef(kwargs);
        Py_DecRef(legend);
      } else {
        PyObject_CallMethod(pyax, "legend", nullptr);
      }
    }
    if (ax.xlim_left) {
      if (ax.is_x_nanotimestamps) {
        auto dt = slow_int64_to_datetime(*ax.xlim_left);
        PyObject_CallMethod(pyax, "set_xlim", "O", dt);
        check_pyerr();
        Py_DecRef(dt);
      } else {
        PyObject_CallMethod(pyax, "set_xlim", "d", static_cast<double>(*ax.xlim_left));
        check_pyerr();
      }
    }
    if (ax.xlim_right) {
      if (ax.is_x_nanotimestamps) {
        auto dt = slow_int64_to_datetime(*ax.xlim_right);
        PyObject_CallMethod(pyax, "set_xlim", "OO", Py_None, dt);
        check_pyerr();
        Py_DecRef(dt);
      } else {
        PyObject_CallMethod(pyax, "set_xlim", "Od", Py_None, static_cast<double>(*ax.xlim_right));
        check_pyerr();
      }
    }

    Py_DecRef(pyax);
  }

  // cleanup
  for (auto pyax : pyaxes) {
    Py_DecRef(pyax);
  }

  PyObject_CallMethod(pyfig, "tight_layout", nullptr);
  check_pyerr();
}

void Matplot::save(const std::experimental::filesystem::path& path) {
  auto ext = path.extension();
  PyObject_CallMethod(pyfig, "savefig", "s", path.c_str());
  check_pyerr();
}
