message(STATUS "<OPS_LIB> Building ops_util under ${CMAKE_CURRENT_SOURCE_DIR}")

AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_SOURCE_DIR} UTIL_SRCS)
add_library(ops_util SHARED ${UTIL_SRCS})

install(TARGETS ops_util DESTINATION lib)
