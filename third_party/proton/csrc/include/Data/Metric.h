#ifndef PROTON_DATA_METRIC_H_
#define PROTON_DATA_METRIC_H_

#include "Utility/Traits.h"
#include <variant>
#include <vector>

namespace proton {

enum class MetricKind { Flexible, Kernel, PCSampling, Count };

using MetricValueType = std::variant<uint64_t, int64_t, double, std::string>;

/// A metric is a class that can be associated with a context.
/// `Metric` is the base class for all metrics.
/// Each `Metric` has a name and a set of values.
/// Each value could be of type `uint64_t`, `int64_t`, or `double`,
/// Each value also has its own name and is either aggregable or not.
/// Currently the only aggregation operation is `addition`.
class Metric {
public:
  Metric(MetricKind kind, size_t size) : kind(kind), values(size) {}

  virtual ~Metric() = default;

  virtual const std::string getName() const = 0;

  virtual const std::string getValueName(int valueId) const = 0;

  virtual bool isAggregable(int valueId) const = 0;

  std::vector<MetricValueType> getValues() const { return values; }

  MetricValueType getValue(int valueId) { return values[valueId]; }

  /// Update a specific value id with the new value.
  void updateValue(int valueId, MetricValueType value) {
    // Handle string and other values separately
    if (std::holds_alternative<std::string>(value)) {
      values[valueId] = std::get<std::string>(value);
    } else {
      std::visit(
          [&](auto &&currentValue, auto &&otherValue) {
            using CurrentType = std::decay_t<decltype(currentValue)>;
            using ValueType = std::decay_t<decltype(otherValue)>;
            if constexpr (std::is_same_v<ValueType, CurrentType>) {
              if (isAggregable(valueId)) {
                currentValue += otherValue;
              } else {
                currentValue = otherValue;
              }
            }
          },
          values[valueId], value);
    }
  }

  /// Update all values of the metric with the same value.
  void updateValue(MetricValueType value) {
    for (int i = 0; i < values.size(); ++i) {
      updateValue(i, value);
    }
  }

  /// Update all values with another metric.
  void updateMetric(Metric &other) {
    for (int i = 0; i < values.size(); ++i) {
      updateValue(i, other.values[i]);
    }
  }

  MetricKind getKind() const { return kind; }

private:
  const MetricKind kind;
  const std::string name;

protected:
  std::vector<MetricValueType> values;
};

/// A flexible metric is provided by users but not the backend profiling API.
/// Each flexible metric has a single value.
/// When aggregable = true, the value can be aggregated over time.
/// When aggregable = false, the value is a property that doesn't change over
/// time.
class FlexibleMetric : public Metric {
public:
  FlexibleMetric(const std::string &valueName,
                 std::variant<MetricValueType> value, bool aggregable)
      : valueName(valueName), Metric(MetricKind::Flexible, 1),
        aggregable(aggregable) {
    std::visit([&](auto &&v) { this->values[0] = v; }, value);
  }

  const std::string getName() const override { return "FlexibleMetric"; }

  const std::string getValueName(int valueId) const override {
    return valueName;
  }

  bool isAggregable(int valueId) const override { return aggregable; }

private:
  const bool aggregable;
  const std::string valueName;
};

class KernelMetric : public Metric {
public:
  enum kernelMetricKind : int {
    StartTime,
    EndTime,
    Invocations,
    Duration,
    DeviceId,
    DeviceType,
    Count,
  };

  KernelMetric() : Metric(MetricKind::Kernel, kernelMetricKind::Count) {}

  KernelMetric(uint64_t startTime, uint64_t endTime, uint64_t invocations,
               uint64_t deviceId, uint64_t deviceType)
      : KernelMetric() {
    this->values[StartTime] = startTime;
    this->values[EndTime] = endTime;
    this->values[Invocations] = invocations;
    this->values[Duration] = endTime - startTime;
    this->values[DeviceId] = deviceId;
    this->values[DeviceType] = deviceType;
  }

  virtual const std::string getName() const { return "KernelMetric"; }

  virtual const std::string getValueName(int valueId) const {
    return VALUE_NAMES[valueId];
  }

  virtual bool isAggregable(int valueId) const { return AGGREGABLE[valueId]; }

private:
  const static inline bool AGGREGABLE[kernelMetricKind::Count] = {
      false, false, true, true, false, false};
  const static inline std::string VALUE_NAMES[kernelMetricKind::Count] = {
      "StartTime (ns)", "EndTime (ns)", "Count",
      "Time (ns)",      "DeviceId",     "DeviceType",
  };
};

class PCSamplingMetric : public Metric {
public:
  enum PCSamplingMetricKind : int {
    NumSamples,
    NumStalledSamples,
    StalledBranchResolving,
    StalledNoInstruction,
    StalledShortScoreboard,
    StalledWait,
    StalledLongScoreboard,
    StalledTexThrottle,
    StalledBarrier,
    StalledMembar,
    StalledIMCMiss,
    StalledMIOThrottle,
    StalledMathPipeThrottle,
    StalledDrain,
    StalledLGThrottle,
    StalledNotSelected,
    StalledMisc,
    StalledDispatchStall,
    StalledSleeping,
    StalledSelected,
    Count,
  };

  PCSamplingMetric()
      : Metric(MetricKind::PCSampling, PCSamplingMetricKind::Count) {}

  PCSamplingMetric(PCSamplingMetricKind kind, uint64_t samples,
                   uint64_t stalledSamples)
      : PCSamplingMetric() {
    this->values[kind] = stalledSamples;
    this->values[PCSamplingMetricKind::NumSamples] = samples;
    this->values[PCSamplingMetricKind::NumStalledSamples] = stalledSamples;
  }

  virtual const std::string getName() const { return "PCSamplingMetric"; }

  virtual const std::string getValueName(int valueId) const {
    return VALUE_NAMES[valueId];
  }

  virtual bool isAggregable(int valueId) const { return true; }

private:
  const static inline std::string VALUE_NAMES[PCSamplingMetricKind::Count] = {
      "NumSamples",
      "NumStalledSamples",
      "StalledBranchResolving",
      "StalledNoInstruction",
      "StalledShortScoreboard",
      "StalledWait",
      "StalledLongScoreboard",
      "StalledTexThrottle",
      "StalledBarrier",
      "StalledMembar",
      "StalledIMCMiss",
      "StalledMIOThrottle",
      "StalledMathPipeThrottle",
      "StalledDrain",
      "StalledLGThrottle",
      "StalledNotSelected",
      "StalledMisc",
      "StalledDispatchStall",
      "StalledSleeping",
      "StalledSelected",
  };
};

} // namespace proton

#endif // PROTON_DATA_METRIC_H_
