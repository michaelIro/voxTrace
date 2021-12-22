#include "PlotAPI.hpp"

void PlotAPI::test(){
    // Create a vector with values from 0 to pi divived into 200 uniform intervals for the x-axis
    sciplot::Vec x = sciplot::linspace(0.0, sciplot::PI, 200);

    // Create a Plot object
    sciplot::Plot plot;

    // Set the x and y labels
    plot.xlabel("x");
    plot.ylabel("y");

    // Set the x and y ranges
    plot.xrange(0.0, sciplot::PI);
    plot.yrange(0.0, 1.0);

    // Set the legend to be on the bottom along the horizontal
    plot.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);

    // Plot sin(i*x) from i = 1 to i = 6
    plot.drawCurve(x, std::sin(1.0 * x)).label("sin(x)");
    plot.drawCurve(x, std::sin(2.0 * x)).label("sin(2x)");
    plot.drawCurve(x, std::sin(3.0 * x)).label("sin(3x)");
    plot.drawCurve(x, std::sin(4.0 * x)).label("sin(4x)");
    plot.drawCurve(x, std::sin(5.0 * x)).label("sin(5x)");
    plot.drawCurve(x, std::sin(6.0 * x)).label("sin(6x)");

    // Show the plot in a pop-up window
    //plot.show();

    // Save the plot to a PDF file
    plot.save("../test-data/plots/example-sine-functions.pdf");
}
