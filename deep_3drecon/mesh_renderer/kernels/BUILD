cc_library(
    name = "rasterize_triangles_impl",
    srcs = ["rasterize_triangles_impl.cc"],
    hdrs = ["rasterize_triangles_impl.h"],
)

cc_test(
    name = "rasterize_triangles_impl_test",
    srcs = ["rasterize_triangles_impl_test.cc"],
    data = [
        "//mesh_renderer/test_data:images",
    ],
    deps = [
        ":rasterize_triangles_impl",
        "//third_party:lodepng",
        "@com_google_googletest//:gtest_main",
    ],
)

genrule(
    name = "rasterize_triangles_kernel",
    srcs = ["rasterize_triangles_grad.cc",
            "rasterize_triangles_op.cc",
            "rasterize_triangles_impl.cc",
            "rasterize_triangles_impl.h"],
    outs = ["rasterize_triangles_kernel.so"],
    cmd = "TF_INC=$$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())');\
           TF_LIB=$$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'); \
           g++ -std=c++11 -shared $(SRCS) -o $@ -fPIC -I$$TF_INC -I$$TF_INC/external/nsync/public -L$$TF_LIB -ltensorflow_framework -O2",
    visibility = ["//mesh_renderer:__subpackages__"],
)
# -D_GLIBCXX_USE_CXX11_ABI=0