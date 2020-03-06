#include "vitro.h"
#include <atomic>
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

template <typename V> PyObject* vector_to_ndarray(const std::vector<V>& vec) {
  Py_intptr_t dims[] = {static_cast<Py_intptr_t>(vec.size())};
  auto arr = PyArray_SimpleNewFromData(1, dims, select_npy_type<V>::type,
                                       const_cast<void*>(reinterpret_cast<const void*>(vec.data())));
  if (arr == nullptr) {
    throw std::runtime_error("cannot convert std::vector to np.ndarray");
  }
  return arr;
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
  }

  ~Timestamps() {
    for (PyObject* datetime : datetimes) {
      Py_DECREF(datetime);
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

Matplot::Matplot(const Figure& fig) {
  make_sure_initialized();

  PyObject* pyplot = PyImport_ImportModule("matplotlib.pyplot");
  if (!pyplot) {
    throw std::runtime_error("Error loading module matplotlib.pyplot");
  }

  // default dpi = 100 = pixels per inch
  // width (px) => width/dpi (inches)
  pyfig = PyObject_CallMethod(pyplot, "figure", "O(dd)", Py_None, fig.width / 100.0, fig.height / 100.0);
  // pyfig = PyObject_CallMethod(pyplot, "figure", nullptr);
  if (pyfig == nullptr) {
    throw std::runtime_error("matplotlib.pyplot.figure() failed");
  }

  if (fig.title.size() > 0) {
    PyObject_CallMethod(pyfig, "suptitle", "s", fig.title.c_str());
  }

  PyObject* result = PyObject_CallMethod(pyfig, "subplots", "iiOOO", fig.nrow, fig.ncol, Py_False, Py_False, Py_False);
  if (result == nullptr) {
    throw std::runtime_error("Figure.subplots() failed");
  }

  std::vector<PyObject*> pyaxes;
  for (Py_intptr_t i = 0; i < fig.nrow; i++) {
    for (Py_intptr_t j = 0; j < fig.ncol; j++) {
      pyaxes.push_back(*reinterpret_cast<PyObject**>(PyArray_GETPTR2(reinterpret_cast<PyArrayObject*>(result), i, j)));
      Py_IncRef(pyaxes.back());
    }
  }
  Py_DecRef(result);

  for (int r = 1; r <= fig.nrow; r++) {
    for (int c = 1; c <= fig.ncol; c++) {
      const auto& ax = fig.readonly_axes(r, c);
      auto* pyax = pyaxes[(r - 1) * fig.ncol + c - 1];
      if (ax.title.size() > 0) {
        PyObject_CallMethod(pyax, "set_title", "s", ax.title.c_str());
      }
      if (ax.xlabel.size() > 0) {
        PyObject_CallMethod(pyax, "set_xlabel", "s", ax.xlabel.c_str());
      }
      if (ax.ylabel.size() > 0) {
        PyObject_CallMethod(pyax, "set_ylabel", "s", ax.ylabel.c_str());
      }

      for (const auto& line : ax.lines) {
        PyObject* x;
        if (ax.is_x_nanotimestamps) {
          Timestamps t(line.xs);
          x = vector_to_ndarray(t.datetimes);
        } else {
          x = vector_to_ndarray(line.xs);
        }
        auto y = vector_to_ndarray(line.ys);

        auto plot = PyObject_GetAttrString(pyax, "plot");
        auto args = Py_BuildValue("(OO)", x, y);
        auto kwargs = Py_BuildValue("{s:s}", "label", line.name.c_str());
        auto pyline = PyObject_Call(plot, args, kwargs);
        if (pyline == nullptr) {
          throw std::runtime_error("cannot draw a line");
        }

        Py_DecRef(x);
        Py_DecRef(y);
        Py_DecRef(args);
        Py_DecRef(kwargs);
        Py_DecRef(plot);
      }
      for (const auto& scatter : ax.scatters) {
        PyObject* x;
        if (ax.is_x_nanotimestamps) {
          Timestamps t(scatter.xs);
          x = vector_to_ndarray(t.datetimes);
        } else {
          x = vector_to_ndarray(scatter.xs);
        }
        auto y = vector_to_ndarray(scatter.ys);

        auto plot = PyObject_GetAttrString(pyax, "scatter");
        auto args = Py_BuildValue("(OO)", x, y);
        auto kwargs = Py_BuildValue("{s:s, s:s}", "label", scatter.name.c_str(), "marker", scatter.marker_type.c_str());
        auto pyscatter = PyObject_Call(plot, args, kwargs);
        if (pyscatter == nullptr) {
          throw std::runtime_error("cannot draw a scatter");
        }

        Py_DecRef(x);
        Py_DecRef(y);
        Py_DecRef(args);
        Py_DecRef(kwargs);
        Py_DecRef(plot);
      }
      for (const auto& area : ax.areas) {
        PyObject* x;
        if (ax.is_x_nanotimestamps) {
          Timestamps t(area.xs);
          x = vector_to_ndarray(t.datetimes);
        } else {
          x = vector_to_ndarray(area.xs);
        }
        auto y1 = vector_to_ndarray(area.y1s);
        auto y2 = vector_to_ndarray(area.y2s);

        auto pyarea = PyObject_CallMethod(pyax, "area", "OOO", x, y1, y2);
        if (pyarea == nullptr) {
          throw std::runtime_error("cannot draw a area");
        }

        Py_DecRef(x);
        Py_DecRef(y1);
        Py_DecRef(y2);
      }
      PyObject_CallMethod(pyax, "legend", nullptr);
      Py_DecRef(pyax);
    }
  }

  // cleanup
  for (auto pyax : pyaxes) {
    Py_DecRef(pyax);
  }
}

void Matplot::save(const std::experimental::filesystem::path& path) {
  auto ext = path.extension();
  if (ext == ".png") {
    PyObject_CallMethod(pyfig, "savefig", "s", path.c_str()); // TODO: kwargs
  } else {
    throw std::runtime_error("unsupported matplotlib export file format: " + ext.string());
  }
}
