#include "vitro.h"

void plot1() {
  Figure fig{};
  fig.title = "t - t - title";

  auto& ax = fig.axes(1, 1);
  ax.line("line1", {1583521630'000'000'000, 1583521730'000'000'000, 1583521830'000'000'000}, {4, 5, 6});
  ax.scatter("scatter2", {1583521630'000'000'000, 1583521730'000'000'000, 1583521830'000'000'000}, {1, 4, 2});
  ax.xlabel = "x-axis-name";
  ax.ylabel = "y-axis-name";
  ax.is_x_nanotimestamps = true;

  Matplot matplot(fig);
  matplot.save("plot1.png");
}

void plot2() {
  Figure fig(2, 2);
  fig.title = "2020-02-20";

  auto& ax1 = fig.axes(2, 1);
  ax1.line("line2", {1, 2, 3}, {4, 5, 6});
  ax1.scatter("scatter5", {4, 5, 6}, {1, 2, 3});
  ax1.title = "row=2 col=1";

  auto& ax2 = fig.axes(1, 2);
  ax2.xlabel = "x-axis-name";
  ax2.ylabel = "y-axis-name";
  ax2.line("line3", {10, 20, 30}, {4, 3, 1});
  ax2.line("line4", {30, 15, 5}, {4, 2, 2.5});
  auto& scatter = ax2.scatter("scatter4", {40, 50, 60, 70}, {2, 5, 0, 10});
  scatter.marker_type = "4";

  Matplot matplot(fig);
  matplot.save("plot2.png");
}

int main() {
  plot1();
  plot2();
  return 0;
}
