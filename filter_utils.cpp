#include "filter_utils.h"

FilterType stringToFilterType(const std::string& str) {
    if (str == "grayscale") return GRAYSCALE;
    if (str == "hdr") return HDR_TONEMAPPING;
    return NONE;
}