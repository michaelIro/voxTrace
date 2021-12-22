//!  API to plotting library (sciplot)
#ifndef PlotAPI_H
#define PlotAPI_H

#include <sciplot/sciplot.hpp>  // namespace sciplot already defined
#include <armadillo>

class PlotAPI{

    public:
        PlotAPI() = delete;
        static void test();
        static void scatter_plot(char* save_path, bool x_right, bool y_up, arma::Mat<double> xy_coordinates);

};

#endif

  