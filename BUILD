load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "vitro",
    srcs = ["vitro.cc"],
    hdrs = ["vitro.h"],
    # srcs = glob(["**/*.cc"]),
    # hdrs = glob(["**/*.h"]),
    copts = [
        "-std=c++17",
    ],
    linkopts = [
        "-lpython3.8",
        "-lstdc++fs",
    ],
)

cc_binary(
    name = "example",
    srcs = ["example.cc"],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":vitro",
    ],
)
