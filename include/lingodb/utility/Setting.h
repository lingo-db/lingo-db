#ifndef LINGODB_UTILITY_SETTING_H
#define LINGODB_UTILITY_SETTING_H
#include <optional>
#include <stdexcept>
#include <string>
namespace utility {

class Setting {
   protected:
   std::string key;

   public:
   Setting(std::string key) : key(key) {
      registerSetting();
   }

   void registerSetting();
   void unregisterSetting();
   void error(std::string message);
   virtual void setValue(std::string value) = 0;
   virtual ~Setting() {
      unregisterSetting();
   }
};
template <class T>
std::optional<T> parseSetting(const std::string& value);

template <>
std::optional<int64_t> parseSetting<int64_t>(const std::string& value);
template <>
std::optional<std::string> parseSetting<std::string>(const std::string& value);
template <>
std::optional<bool> parseSetting<bool>(const std::string& value);
template <>
std::optional<double> parseSetting<double>(const std::string& value);

template <class T>
class GlobalSetting : public Setting {
   T value;

   public:
   GlobalSetting(std::string key, T defaultValue) : Setting(key), value(std::move(defaultValue)) {
      initialize();
   }
   void initialize() {
      std::string envName = "LINGODB_";
      envName.reserve(envName.size() + key.size());
      std::string keyToProcess = key;
      if (key.starts_with("system.")) {
         keyToProcess = key.substr(7);
      }
      for (auto& c : keyToProcess) {
         if (c == '.') {
            envName.push_back('_');
         } else {
            envName.push_back(std::toupper(c));
         }
      }
      char* envValue = std::getenv(envName.c_str());
      if (envValue) {
         setValue(envValue);
      }
   }

   void setValue(std::string value) override {
      auto parsed = parseSetting<T>(value);
      if (parsed) {
         this->value = *parsed;
      } else {
         error("Invalid value for setting " + key);
      }
   }

   T getValue() const { return value; }

   ~GlobalSetting() {
   }
};

void setSetting(std::string key, std::string value);
} // namespace utility
#endif //LINGODB_UTILITY_SETTING_H
