#include "vitro.h"

void plot1() {
  Figure fig{};
  fig.title = "t - t - title";

  {
    auto& ax = fig.axes(1, 1);
    ax.line("line1", {1583521630'000'000'000, 1583521730'000'000'000, 1583521830'000'000'000}, {4, 5, 6});
    ax.scatter("scatter2", {1583521630'000'000'000, 1583521730'000'000'000, 1583521830'000'000'000}, {1, 4, 2});
    ax.text("This is vitro.\n1+1=3", 0.9, 0.9);
    ax.text("left bottom", 0.1, 0.1);
    ax.text("right mid", 0.9, 0.5);
    ax.text("right bottom", 0.9, 0.1);

    ax.xlabel = "x-axis-name";
    ax.ylabel = "y-axis-name";
    ax.is_x_nanotimestamps = true;
  }

  {
    auto& ax = fig.twinx(1, 1);
    ax.line("line2", {1583521660'000'000'000, 1583521710'000'000'000, 1583521780'000'000'000}, {4, 2, 3.5});
    ax.is_x_nanotimestamps = true;
  }

  Matplot matplot(fig);
  matplot.save("plot1.png");
}

void plot2() {
  Figure fig(2, 2);
  fig.title = "2020-02-20";

  {
    auto& ax = fig.axes(2, 1);
    ax.title = "row=2 col=1";
    ax.line("line2", {1, 2, 3}, {4, 5, 6});
    auto& scatter = ax.scatter("scatter5", {4, 5, 6}, {1, 2, 3});
    scatter.marker_size = 100;
    scatter.color = "green";
    scatter.marker_face_color = "none";
  }

  {
    auto& ax = fig.axes(1, 2);
    ax.xlabel = "x-axis-name";
    ax.ylabel = "y-axis-name";
    ax.line("line3", {10, 20, 30}, {4, 3, 1});
    ax.line("line4", {30, 15, 5}, {4, 2, 2.5});
    auto& scatter = ax.scatter("scatter4", {40, 50, 60, 70}, {2, 5, 0, 10});
    scatter.marker_type = '4';
    scatter.marker_size = 200;
  }

  {
    auto& ax = fig.axes(2, 2);
    ax.line("log-scale-x", {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5});
    ax.xscale = "log";
  }

  Matplot matplot(fig);
  matplot.save("plot2.png");
}

void plot3() {
  Figure fig(2, 1);
  fig.title = "area chart";
  fig.height_ratios = {3, 1};
  {
    auto& ax = fig.axes(1, 1);
    ax.area("large-area-series", {1, 2, 3}, {4, 5, 6}, {6, 6, 4});
  }
  {
    auto& ax = fig.axes(2, 1);
    ax.area("small-area-series", {10, 25, 30, 5}, {46, 5, 6, 10}, {6, 6, 4, 5});
  }

  Matplot matplot(fig);
  matplot.save("plot3.png");
}

void plot4() {
  Figure fig(3, 1);
  fig.title = "histogram";

  std::vector<double> x1;
  unsigned int k = 0;
  for (int i = 0; i < 1000; i++) {
    k = (k + 9832103711) % 217;
    double s = k;
    k = (k + 9832103711) % 217;
    s += k;
    k = (k + 9832103711) % 217;
    s += k;
    k = (k + 9832103711) % 217;
    s += k;
    x1.push_back(s / 4 / 217.0);
  }
  std::vector<double> x2;
  for (int i = 0; i < 1000; i++) {
    double s = 0;
    k = (k + 2103713) % 981;
    s += k;
    k = (k + 2103713) % 981;
    s += k;
    k = (k + 2103713) % 981;
    s += k;
    k = (k + 2103713) % 981;
    s += k;
    x2.push_back(s / 5 / 981);
  }

  std::vector<double> x3;
  for (int i = 0; i < 1000; i++) {
    k = (k + 98323711) % 137;
    double s = k;
    k = (k + 98323711) % 137;
    s += k;
    k = (k + 98323711) % 137;
    s += k;
    k = (k + 98323711) % 137;
    s += k;
    x3.push_back(s / 4 / 100.0);
  }
  std::vector<double> x4;
  for (int i = 0; i < 1000; i++) {
    double s = 0;
    k = (k + 11210713) % 9817;
    s += k;
    k = (k + 11210713) % 9817;
    s += k;
    k = (k + 11210713) % 9817;
    s += k;
    k = (k + 11210713) % 9817;
    s += k;
    x4.push_back(s / 5 / 10000);
  }

  {
    auto& ax = fig.axes(1, 1);

    {
      auto& hist = ax.histogram({"h1", "h2"}, 40, {x1, x2});
      hist.normalize_area = true;
    }
    {
      auto& hist = ax.histogram({"h3", "h4"}, 40, {x3, x4});
      hist.type = "step";
      hist.normalize_area = true;
    }
  }
  {
    auto& ax = fig.axes(2, 1);
    ax.title = "log bins";
    ax.xscale = "log";
    ax.legend = "upper left";
    {
      auto& hist = ax.histogram({"h1", "h2"}, 40, {x1, x2});
      hist.bin_log_scale = true;
    }
    {
      auto& hist = ax.histogram({"h3", "h4"}, 40, {x3, x4});
      hist.type = "step";
      hist.bin_log_scale = true;
    }
  }
  {
    auto& ax = fig.axes(3, 1);

    {
      auto& hist = ax.histogram({"h1", "h2"}, 40, {x1, x2});
      hist.normalize_area = true;
      hist.cumulative = 1;
    }
    {
      auto& hist = ax.histogram({"h3", "h4"}, 40, {x3, x4});
      hist.normalize_area = true;
      hist.type = "step";
      hist.cumulative = -1;
    }
  }

  Matplot matplot(fig);
  matplot.save("plot4.png");
}

int main() {
  plot1();
  plot2();
  plot3();
  plot4();
  return 0;
}
