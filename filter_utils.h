#ifndef FILTER_UTILS_H
#define FILTER_UTILS_H

#include <string>

enum FilterType {
    NONE,
    GRAYSCALE,
    HDR_TONEMAPPING // new hdr 
};

FilterType stringToFilterType(const std::string& str);

#endif