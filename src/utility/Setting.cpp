#include "lingodb/utility/Setting.h"
#include <algorithm>
#include <mutex>
#include <unordered_map>
template <>
std::optional<int64_t> utility::parseSetting<int64_t>(const std::string& value) {
   size_t charactersParsed;
   int64_t res = std::stoll(value, &charactersParsed);
   if (charactersParsed != value.size()) {
      return std::nullopt;
   }
   return res;
}
template<>
std::optional<std::string> utility::parseSetting<std::string>(const std::string& value) {
   return value;
}
template<>
std::optional<bool> utility::parseSetting<bool>(const std::string& value) {
   std::string mutValue=value;
   std::transform(mutValue.begin(), mutValue.end(), mutValue.begin(), ::tolower);
   if (mutValue == "true" ) {
      return true;
   } else if (mutValue == "false") {
      return false;
   } else {
      try {
         size_t charactersParsed;
         int res = std::stoi(value, &charactersParsed);
         if (charactersParsed != value.size()) {
            return std::nullopt;
         }
         return res;
      } catch (std::invalid_argument& e) {
         return std::nullopt;
      }
   }
}
template<>
std::optional<double> utility::parseSetting<double>(const std::string& value) {
   size_t charactersParsed;
   double res = std::stod(value, &charactersParsed);
   if (charactersParsed != value.size()) {
      return std::nullopt;
   }
   return res;
}
struct RegisteredSettings{
   std::mutex mutex;
   std::unordered_map<std::string, utility::Setting*> settings;
};
namespace{
RegisteredSettings& getRegisteredSettings(){
   static RegisteredSettings settings;
   return settings;
}
} // namespace


void utility::Setting::registerSetting() {
   auto& settings = getRegisteredSettings();
   std::lock_guard<std::mutex> lock(settings.mutex);
   if (settings.settings.find(key) != settings.settings.end()) {
      throw std::runtime_error("Setting with key " + key + " already exists");
   }
   settings.settings[key] = this;
}

void utility::Setting::unregisterSetting() {
   auto& settings = getRegisteredSettings();
   std::lock_guard<std::mutex> lock(settings.mutex);
   settings.settings.erase(key);
}
void utility::Setting::error(std::string message) {
   throw std::runtime_error(message);
}

void utility::setSetting(std::string key, std::string value) {
   auto& settings = getRegisteredSettings();
   std::lock_guard<std::mutex> lock(settings.mutex);
   auto it = settings.settings.find(key);
   if (it == settings.settings.end()) {
      throw std::runtime_error("Setting with key " + key + " does not exist");
   }
   it->second->setValue(value);
}

