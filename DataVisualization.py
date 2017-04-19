import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class DataVisualization():
    x = []
    y = []
    # Graph Colors
    color_pallet = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    def init(self):
        print("Something")
        #self.scale_colors()


    def scale_colors(self):
        import struct
        # Scaling RGB values to [0.1] for matplotlib
        for i in range(len(self.color_pallet)):
            rgb = self.color_pallet[i]
            self.color_pallet[i] = "#"+struct.pack('BBB',*rgb).encode('hex')

    def scatter_plot(self):
        print("Sala")

    def bar_chart(self,save = False):
        # Due to an agreement with the ChessGames.com admin, I cannot make the data
        # for this plot publicly available. This function reads in and parses the
        # chess data set into a tabulated pandas DataFrame.
        import random
        data = random.sample(range(1, 100000), 100)
        data2 = random.sample(range(1, 200000), 100)

        print(self.color_pallet)

        ax, plt = self.set_plot_properties()

        # Plot the histogram. Note that all I'm passing here is a list of numbers.
        # matplotlib automatically counts and bins the frequencies for us.
        # "#3F5D7D" is the nice dark blue color.
        # Make sure the data is sorted into enough bins so you can see the distribution.
        from random import randint
        plt.hist(list(data) + list(data2),
                 color=self.color_pallet[randint(0,len(self.color_pallet)-1)], bins=100)

        # Always include your data source(s) and copyright notice! And for your
        # data sources, tell your viewers exactly where the data came from,
        # preferably with a direct link to the data. Just telling your viewers
        # that you used data from the "U.S. Census Bureau" is completely useless:
        # the U.S. Census Bureau provides all kinds of data, so how are your
        # viewers supposed to know which data set you used?
        plt.text(1300, -5000, "DSGA3003 Project |\
                              sm7029 | avm358 | pc2310", fontsize=10)
        plt.show()
        #if(save):
        plt.savefig("images/"+str(time.time())+"line_.png", bbox_inches="tight");

    def linear_regression(self):
        print("Regress")
        n = 50
        x = np.random.randn(n)
        y = x * np.random.randn(n)
        fig, ax = plt.subplots()
        #ax, fig = self.set_plot_properties(ax, fig)
        fit = np.polyfit(x, y, deg=1)
        ax.plot(x, fit[0] * x + fit[1], color='red')
        ax.scatter(x, y)
        fig.show()

    def line_curve(self,caption = None, heading = None):
        # Read the data into a pandas DataFrame.
        data = []
        plt.figure(figsize=(12, 14))

        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.ylim(0, 90)
        plt.xlim(1968, 2014)


        plt.yticks(range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)], fontsize=14)
        plt.xticks(fontsize=14)

        for y in range(10, 91, 10):
            plt.plot(range(1968, 2012), [y] * len(range(1968, 2012)), "--", lw=0.5, color="black", alpha=0.3)

        plt.tick_params(axis="both", which="both", bottom="off", top="off",
                        labelbottom="on", left="off", right="off", labelleft="on")

        majors = ['Health Professions', 'Public Administration', 'Education', 'Psychology',
                  'Foreign Languages', 'English', 'Communications\nand Journalism',
                  'Art and Performance', 'Biology', 'Agriculture',
                  'Social Sciences and History', 'Business', 'Math and Statistics',
                  'Architecture', 'Physical Sciences', 'Computer Science',
                  'Engineering']

        for rank, column in enumerate(majors):
            # Plot each line separately with its own color, using the Tableau 20
            # color set in order.
            plt.plot(data.Year.values,
                     data[column.replace("\n", " ")].values,
                     lw=2.5, color=self.color_pallet[rank])

            # Add a text label to the right end of every line. Most of the code below
            # is adding specific offsets y position because some labels overlapped.
            y_pos = data[column.replace("\n", " ")].values[-1] - 0.5
            if column == "Foreign Languages":
                y_pos += 0.5
            elif column == "English":
                y_pos -= 0.5
            elif column == "Communications\nand Journalism":
                y_pos += 0.75
            elif column == "Art and Performance":
                y_pos -= 0.25
            elif column == "Agriculture":
                y_pos += 1.25
            elif column == "Social Sciences and History":
                y_pos += 0.25
            elif column == "Business":
                y_pos -= 0.75
            elif column == "Math and Statistics":
                y_pos += 0.75
            elif column == "Architecture":
                y_pos -= 0.75
            elif column == "Computer Science":
                y_pos += 0.75
            elif column == "Engineering":
                y_pos -= 0.25

            plt.text(2011.5, y_pos, column, fontsize=14, color=self.color_pallet[rank])

        # matplotlib's title() call centers the title on the plot, but not the graph,
        # Here used text() call to customize where the title goes.
        if heading is not None:
            plt.text(1995, 93, heading, fontsize=17, ha="center")

        plt.text(1966, -8, "\nDSGA3110 Project"
                           "\nsm7029 | avm358 | pc2310", fontsize=10)

        plt.savefig(str(time.time())+"_line.png", bbox_inches="tight")
        plt.show()


    def set_plot_properties(self,ax,plt):

        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.figure(figsize=(12, 9))

        plt.xticks(fontsize=14)
        plt.yticks(range(5000, 30001, 5000), fontsize=14)


        plt.xlabel("Elo Rating", fontsize=15)
        plt.ylabel("Count", fontsize=15)
        return ax,plt



d = DataVisualization()
#d.linear_regression()
d.line_curve()