include (${CMAKE_SOURCE_DIR}/unittest/googletest.cmake)

include(GoogleTest)
enable_testing()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

function(add_triton_ut)
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs SRCS LIBS DEFS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_test(NAME ${__NAME}
          COMMAND ${__NAME})
  add_executable(
          ${__NAME}
          ${__SRCS})
  target_link_libraries(
          ${__NAME}
          PRIVATE
          GTest::gtest_main
          ${triton_libs}
          ${dialect_libs}
          ${conversion_libs}
          gmock
          ${__LIBS})

  target_compile_options(${__NAME} PRIVATE -fno-rtti)

  target_compile_definitions(${__NAME} PRIVATE ${__DEFS})

#   Without the TEST_DISCOVERY_TIMEOUT, the tests randomly time out on my mac
#   laptop.  I think the issue may be that the very first time you run a program
#   it's a bit slow.
#   gtest_discover_tests(${__NAME} PROPERTIES TEST_DISCOVERY_TIMEOUT 60)
endfunction()

add_subdirectory(Conversion)
