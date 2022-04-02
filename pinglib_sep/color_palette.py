'''
    This code is a modified version of PyColorPalette (https://github.com/Dilan1020/PyColorPalette)
    We add a background criterion to neglect those part in patches from whole slide image
'''

import numpy as np
from PIL import Image
import re
import urllib.request
import io
from functools import partial
from .imgprocessing import rgb_threshold

class ColorPalette():

    def __init__(self, path, k=5, show_clustering=False, foreground=None):
        self.foreground = foreground
        if k > 5:
            raise ValueError("Maximum value for k is 5")
        self.k = k
        self.path = path
        self.show_clustering = show_clustering
        if isinstance(path, np.ndarray):
            urls = None
        else:
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', path)

        if not urls:
            self.__is_url = False
        else:
            self.__is_url = True

        self.__color_palette()

    def __color_palette(self):
        if self.__is_url:
            with urllib.request.urlopen(self.path) as url:
                f = io.BytesIO(url.read())
            im = Image.open(f).convert('RGB')
        elif isinstance(self.path, np.ndarray):
            #   增加的两行语句，使其可以处理ndarray
            im = Image.fromarray(self.path).convert('RGB')
        else:
            im = Image.open(self.path).convert('RGB')

        pixels = im.load()
        width, height = im.size

        pixel_dict = dict()
        _total_pixels = 0

        #   Newly added script, for filtering out backgrounds
        if self.foreground is not None:
            with_foreground = True

            if isinstance(self.foreground, tuple) or isinstance(self.foreground, list):
                valid_map = rgb_threshold(np.array(im),
                                          r_th=self.foreground[0],
                                          g_th=self.foreground[1],
                                          b_th=self.foreground[2], )
            else:
                valid_map = self.foreground
        else:
            with_foreground = False
            valid_map = None

        if with_foreground:
            for col in range(width):
                for row in range(height):
                    cpixel = pixels[col, row]
                    if not valid_map[row, col]:
                        continue
                    _total_pixels += 1
                    if cpixel in pixel_dict:
                        pixel_dict[cpixel] = pixel_dict[cpixel] + 1
                    else:
                        pixel_dict[cpixel] = 1
        else:
            for col in range(width):
                for row in range(height):
                    cpixel = pixels[col, row]
                    _total_pixels += 1
                    if cpixel in pixel_dict:
                        pixel_dict[cpixel] = pixel_dict[cpixel] + 1
                    else:
                        pixel_dict[cpixel] = 1

        self.pixel_dict = pixel_dict
        self._total_pixels = _total_pixels
        sorted_tups = [(k, float(pixel_dict[k] / _total_pixels) * 100) for k in
                       sorted(pixel_dict, key=pixel_dict.get, reverse=True)]

        # Getting estimated colors
        k_rgb = Kmeans(k=self.k, show_clustering=self.show_clustering).run(im)

        sorted_tups = self.__cross_reference(sorted_tups, k_rgb)
        sorted_tups = [e for e in sorted_tups if e[1] > 1]

        self.sorted_tups = sorted_tups
        self.dict = pixel_dict

    def __cross_reference(self, all_colors, estimated_colors):
        new_sorted_tups = dict()
        for curr_color in all_colors:
            closestColor = min(estimated_colors, key=partial(self.__color_difference, curr_color[0]))
            if closestColor in new_sorted_tups:
                new_sorted_tups[closestColor] = new_sorted_tups[closestColor] + float(
                    self.pixel_dict[curr_color[0]] / self._total_pixels) * 100
            else:
                new_sorted_tups[closestColor] = float(self.pixel_dict[curr_color[0]] / self._total_pixels) * 100
        return [(k, new_sorted_tups[k]) for k in sorted(new_sorted_tups, key=new_sorted_tups.get, reverse=True)]

    def __color_difference(self, testColor, otherColor):
        difference = 0
        try:
            difference += abs(testColor[0] - otherColor[0])
            difference += abs(testColor[1] - otherColor[1])
            difference += abs(testColor[2] - otherColor[2])
        except Exception as e:
            print("Error on color: {}\nError: {}".format(testColor, e.args))

        return difference

    def _rgb2hex(self, rgb_tuple):
        r, g, b = rgb_tuple
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def __re_round(self, li):
        try:
            return int(round(li, 0))
        except TypeError:
            return type(li)(self.__re_round(x) for x in li)

    def get_top_colors(self, n=5, ratio=False, rounded=True, to_hex=False):
        if n > 5:
            raise ValueError("Max query is 5")

        sorted_tups = self.sorted_tups

        if rounded:
            sorted_tups = self.__re_round(sorted_tups)

        if not ratio:
            if to_hex:
                sorted_tups = self.__re_round(sorted_tups)
                hex_tups = [self._rgb2hex(f[0]) for f in sorted_tups[:n]]
                return hex_tups
            else:
                return [color[0] for color in sorted_tups[:n]]
        else:
            if to_hex:
                sorted_tups = self.__re_round(sorted_tups)
                hex_tups = [(self._rgb2hex(f[0]), f[1]) for f in sorted_tups[:n]]
                return hex_tups
            else:
                return [color for color in sorted_tups[:n]]

    def get_color(self, index, ratio=False, rounded=True, to_hex=False):
        if index > 5:
            raise ValueError("Max query is 5")

        sorted_tups = self.sorted_tups

        if rounded:
            sorted_tups = self.__re_round(sorted_tups)

        if not ratio:
            if to_hex:
                sorted_tups = self.__re_round(sorted_tups)
                return self._rgb2hex(sorted_tups[index][0])
            else:
                return sorted_tups[index][0]
        else:
            if to_hex:
                sorted_tups = self.__re_round(sorted_tups)
                val = self._rgb2hex(sorted_tups[index][0])
                ratio = sorted_tups[index][1]
                return (val, ratio)
            else:
                return sorted_tups[index]


'''
Modified Version of ZeevG's K-means implementation
'''

import numpy
import random
from PIL import Image


class Cluster(object):

    def __init__(self):
        self.pixels = []
        self.centroid = None

    def addPoint(self, pixel):
        self.pixels.append(pixel)

    def setNewCentroid(self):
        R = [colour[0] for colour in self.pixels]
        G = [colour[1] for colour in self.pixels]
        B = [colour[2] for colour in self.pixels]

        R = sum(R) / (len(R) + 1)
        G = sum(G) / (len(G) + 1)
        B = sum(B) / (len(B) + 1)

        self.centroid = (R, G, B)
        self.pixels = []

        return self.centroid


class Kmeans(object):

    def __init__(self, k=3, max_iterations=5, min_distance=5.0, size=200, show_clustering=False):
        self.k = k
        self.max_iterations = max_iterations
        self.min_distance = min_distance
        self.show_clustering = show_clustering
        self.size = (size, size)

    def run(self, image):
        self.image = image
        self.image.thumbnail(self.size)
        self.pixels = numpy.array(list(image.getdata()), dtype=numpy.uint8)

        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None
        randomPixels = []
        while len(randomPixels) < 5:
            randomChoice = tuple(random.choice(list(self.pixels)))
            while randomChoice in randomPixels:
                randomChoice = tuple(random.choice(list(self.pixels)))
            randomPixels.append(randomChoice)
        randomPixels = [list(elem) for elem in randomPixels]

        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]

        iterations = 0

        while self.shouldExit(iterations) is False:

            self.oldClusters = [cluster.centroid for cluster in self.clusters]

            # print(iterations)

            for pixel in self.pixels:
                self.assignClusters(pixel)

            for cluster in self.clusters:
                cluster.setNewCentroid()

            iterations += 1
        if self.show_clustering:
            self.showClustering()
        return [cluster.centroid for cluster in self.clusters]

    def assignClusters(self, pixel):
        shortest = float('Inf')
        for cluster in self.clusters:
            distance = self.calcDistance(cluster.centroid, pixel)
            if distance < shortest:
                shortest = distance
                nearest = cluster

        nearest.addPoint(pixel)

    def calcDistance(self, a, b):

        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    def shouldExit(self, iterations):

        if self.oldClusters is None:
            return False
        for idx in range(self.k):
            dist = self.calcDistance(
                numpy.array(self.clusters[idx].centroid),
                numpy.array(self.oldClusters[idx])
            )
            if dist < self.min_distance:
                return True
        if iterations <= self.max_iterations:
            return False

        return True

    def showClustering(self):

        localPixels = [None] * len(self.image.getdata())

        for idx, pixel in enumerate(self.pixels):
            shortest = float('Inf')
            for cluster in self.clusters:
                distance = self.calcDistance(
                    cluster.centroid,
                    pixel
                )
                if distance < shortest:
                    shortest = distance
                    nearest = cluster

            localPixels[idx] = nearest.centroid

        w, h = self.image.size
        localPixels = numpy.asarray(localPixels) \
            .astype('uint8') \
            .reshape((h, w, 3))

        colourMap = Image.fromarray(localPixels)
        colourMap.show()


'''


Copyright (c) 2014, Ze'ev Gilovitz All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


'''
