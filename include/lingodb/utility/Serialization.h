#ifndef LINGODB_UTILITY_SERIALIZATION_H
#define LINGODB_UTILITY_SERIALIZATION_H
#include <cstddef>
#include <fstream>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <vector>
namespace lingodb::utility {
struct ByteWriter {
   virtual void write(const std::byte* data, size_t size) = 0;
   template <class T>
   void write(const T& t) {
      static_assert(std::is_standard_layout<T>::value, "T must be standard layout type");
      write(reinterpret_cast<const std::byte*>(&t), sizeof(T));
   }
   virtual ~ByteWriter() = default;
};
struct ByteReader {
   virtual void read(std::byte* data, size_t size) = 0;
   template <class T>
   T read() {
      static_assert(std::is_standard_layout<T>::value, "T must be standard layout type");
      T t;
      read(reinterpret_cast<std::byte*>(&t), sizeof(T));
      return t;
   }
   virtual ~ByteReader() = default;
};
using marker_t = uint16_t;
constexpr static marker_t present = 0xFFFC;
constexpr static marker_t notPresent = 0xFFFD;

constexpr static marker_t objectStart = 0xFFFE;
constexpr static marker_t objectEnd = 0xFFFF;
//concept for checking for unique ptr
template <class T>
concept IsUniquePtr = requires(T& t) {
   typename T::element_type; // Ensure T has an `element_type`
   requires std::same_as<T, std::unique_ptr<typename T::element_type>>;
};
//concept for checking for shared ptr
template <class T>
concept IsSharedPtr = requires(T& t) {
   typename T::element_type; // Ensure T has an `element_type`
   requires std::same_as<T, std::shared_ptr<typename T::element_type>>;
};
//concept for checking for std::vector
template <class T>
concept IsVector = requires(T& t) {
   typename T::value_type; // Ensure T has an `value_type`
   requires std::same_as<T, std::vector<typename T::value_type>>;
};

template <class T>
concept IsPair = requires(T& t) {
   typename T::first_type; // Ensure T has an `first_type`
   typename T::second_type; // Ensure T has an `second_type`
   requires std::same_as<T, std::pair<typename T::first_type, typename T::second_type>>;
};
//concept for checking for enum class with valid underlying type (uint8_t or uint16_t)
template <typename T>
concept EnumClassWithValidUnderlyingType =
   std::is_enum_v<T> &&
   !std::is_convertible_v<T, int> && // Ensures it's a scoped enum (enum class)
   (std::is_same_v<std::underlying_type_t<T>, uint8_t> ||
    std::is_same_v<std::underlying_type_t<T>, uint16_t>);
//concept for checking for optional
template <class T>
concept IsOptional = requires(T& t) {
   typename T::value_type; // Ensure T has an `value_type`
   requires std::same_as<T, std::optional<typename T::value_type>>;
};
//concept for checking for std::unordered_nmap
template <class T>
concept IsUnorderedMap = requires(T& t) {
   typename T::mapped_type; // Ensure T has an `value_type`
   typename T::key_type; // Ensure T has an `key_type`
   requires std::same_as<T, std::unordered_map<typename T::key_type, typename T::mapped_type, typename T::hasher, typename T::key_equal>>;
};
class Serializer {
   ByteWriter& writer;

   public:
   Serializer(ByteWriter& writer) : writer(writer) {}

   private:
   void writeValue(bool value) {
      writer.write(value);
   }
   void writeValue(uint8_t value) {
      writer.write(value);
   }
   void writeValue(uint16_t value) {
      writer.write(value);
   }
   void writeValue(int value) {
      writer.write(value);
   }
   void writeValue(float value) {
      writer.write(value);
   }
   void writeValue(double value) {
      writer.write(value);
   }
   void writeValue(size_t value) {
      writer.write(value);
   }
   void writeValue(int64_t value) {
      writer.write(value);
   }
   void writeValue(const std::string_view& value) {
      writeValue(value.length());
      writer.write(reinterpret_cast<const std::byte*>(value.data()), value.length());
   }
   template <class T>
   void writeValue(const std::unique_ptr<T>& value) {
      if (value) {
         writeValue(present);
         writeValue(*value);
      } else {
         writeValue(notPresent);
      }
   }
   template <class T>
   void writeValue(const std::shared_ptr<T>& value) {
      if (value) {
         writeValue(present);
         writeValue(*value);
      } else {
         writeValue(notPresent);
      }
   }
   template <class T>
   void writeValue(const std::vector<T>& value) {
      writeValue(value.size());
      for (const auto& v : value) {
         writeValue(v);
      }
   }
   template <class K, class V, class H, class E>
   void writeValue(const std::unordered_map<K, V, H, E>& value) {
      writeValue(value.size());
      for (const auto& [k, v] : value) {
         writeValue(k);
         writeValue(v);
      }
   }
   template <class T>
   void writeValue(const std::optional<T>& value) {
      if (value.has_value()) {
         writeValue(present);
         writeValue(*value);
      } else {
         writeValue(notPresent);
      }
   }
   template <class U, class V>
   void writeValue(const std::pair<U, V>& value) {
      writeValue(value.first);
      writeValue(value.second);
   }
   template <EnumClassWithValidUnderlyingType E>
   void writeValue(E e) {
      using Underlying = std::underlying_type_t<E>;
      if constexpr (std::is_same_v<Underlying, uint8_t>) {
         writeValue(static_cast<uint8_t>(e));
      } else if constexpr (std::is_same_v<Underlying, uint16_t>) {
         writeValue(static_cast<uint16_t>(e));
      }
   }
   void startObject() {
      writeValue(objectStart);
   }
   void endObject() {
      writeValue(objectEnd);
   }
   template <class T>
      requires requires(const T& t, Serializer& self) { t.serialize(self); }
   void writeValue(const T& value) {
      startObject();
      value.serialize(*this);
      endObject();
   }

   public:
   template <class T>
   void writeProperty(marker_t propertyId, const T& value) {
      writer.write(propertyId);
      writeValue(value);
      writer.write(propertyId);
   }
};
class Deserializer {
   ByteReader& reader;

   public:
   Deserializer(ByteReader& reader) : reader(reader) {}

   private:
   template <std::same_as<bool> T>
   T read() {
      return reader.read<T>();
   }
   template <std::same_as<uint8_t> T>
   T read() {
      return reader.read<T>();
   }
   template <std::same_as<uint16_t> T>
   T read() {
      return reader.read<T>();
   }
   template <std::same_as<int> T>
   T read() {
      return reader.read<T>();
   }
   template <std::same_as<float> T>
   T read() {
      return reader.read<T>();
   }
   template <std::same_as<double> T>
   T read() {
      return reader.read<T>();
   }
   template <std::same_as<size_t> T>
   T read() {
      return reader.read<T>();
   }
   template <std::same_as<int64_t> T>
   T read() {
      return reader.read<T>();
   }
   template <std::same_as<std::string> T>
   T read() {
      size_t length = read<size_t>();
      std::string str;
      str.resize(length);
      reader.read(reinterpret_cast<std::byte*>(str.data()), length);
      return str;
   }
   template <IsUniquePtr T>
      requires requires(T& t, Deserializer& self) {
         typename T::element_type;
         requires std::same_as<typename T::element_type, decltype(T::element_type::deserialize(self))>;
      }
   T read() {
      if (read<marker_t>() == notPresent) {
         return nullptr;
      }
      return std::make_unique<typename T::element_type>(read<typename T::element_type>());
   }
   template <IsUniquePtr T>
      requires requires(T& t, Deserializer& self) {
         typename T::element_type;
         requires std::same_as<T, decltype(T::element_type::deserialize(self))>;
      }
   T read() {
      if (read<marker_t>() == notPresent) {
         return nullptr;
      }
      startObject();

      auto res = T::element_type::deserialize(*this);
      endObject();
      return res;
   }
   template <IsSharedPtr T>
      requires requires(T& t, Deserializer& self) {
         typename T::element_type;
         requires std::same_as<typename T::element_type, decltype(T::element_type::deserialize(self))>;
      }
   T read() {
      if (read<marker_t>() == notPresent) {
         return nullptr;
      }
      return std::make_shared<typename T::element_type>(read<typename T::element_type>());
   }
   template <IsSharedPtr T>
      requires requires(T& t, Deserializer& self) {
         typename T::element_type;
         requires std::same_as<T, decltype(T::element_type::deserialize(self))>;
      }
   T read() {
      if (read<marker_t>() == notPresent) {
         return nullptr;
      }
      startObject();
      auto res = T::element_type::deserialize(*this);
      endObject();
      return res;
   }
   template <IsVector T>
   T read() {
      size_t length = read<size_t>();
      T vec;
      vec.reserve(length);
      for (size_t i = 0; i < length; i++) {
         vec.push_back(read<typename T::value_type>());
      }
      return vec;
   }
   template <EnumClassWithValidUnderlyingType E>
   E read() {
      using Underlying = std::underlying_type_t<E>;
      if constexpr (std::is_same_v<Underlying, uint8_t>) {
         return static_cast<E>(read<uint8_t>());
      } else if constexpr (std::is_same_v<Underlying, uint16_t>) {
         return static_cast<E>(read<uint16_t>());
      }
   }
   template <IsOptional T>
   T read() {
      if (read<marker_t>() == notPresent) {
         return std::nullopt;
      }
      return read<typename T::value_type>();
   }
   template <IsPair T>
   T read() {
      return {read<typename T::first_type>(), read<typename T::second_type>()};
   }
   template <IsUnorderedMap T>
   T read() {
      size_t length = read<size_t>();
      T res;
      for (size_t i = 0; i < length; i++) {
         res.insert({read<typename T::key_type>(), read<typename T::mapped_type>()});
      }
      return res;
   }

   void startObject();
   void endObject();

   void startProperty(marker_t propertyId);
   void endProperty(marker_t propertyId);
   template <class T>
      requires requires(Deserializer& self) { T::deserialize(self); }
   T read() {
      startObject();
      T t = T::deserialize(*this);
      endObject();
      return t;
   }

   public:
   template <class T>
   T readProperty(marker_t propertyId) {
      startProperty(propertyId);
      T t = read<T>();
      endProperty(propertyId);
      return t;
   }
};
class SimpleByteWriter : public ByteWriter {
   std::vector<std::byte> buffer;

   public:
   void write(const std::byte* data, size_t size) override {
      buffer.insert(buffer.end(), data, data + size);
   }
   const std::byte* data() const {
      return buffer.data();
   }
   size_t size() const {
      return buffer.size();
   }
};
class SimpleByteReader : public ByteReader {
   const std::byte* buffer;
   size_t size;

   public:
   SimpleByteReader(const std::byte* buffer, size_t size) : buffer(buffer), size(size) {}
   void read(std::byte* data, size_t size) override;
};
class FileByteWriter : public ByteWriter {
   std::ofstream ostream;

   public:
   FileByteWriter(std::string filename);
   void write(const std::byte* data, size_t size) override;
   ~FileByteWriter() = default;
};
class FileByteReader : public ByteReader {
   std::ifstream istream;

   public:
   FileByteReader(std::string filename);
   void read(std::byte* data, size_t size) override;
   ~FileByteReader() = default;
};

template <class T>
T deserializeFromHexString(std::string_view hexString) {
   std::vector<std::byte> bytes;
   bytes.reserve(hexString.size() / 2);
   for (size_t i = 0; i < hexString.size(); i += 2) {
      std::string_view byteString = hexString.substr(i, 2);
      bytes.push_back(static_cast<std::byte>(std::stoi(std::string(byteString), nullptr, 16)));
   }
   SimpleByteReader reader(bytes.data(), bytes.size());
   Deserializer deserializer(reader);
   return deserializer.readProperty<T>(0);
}

template <class T>
std::string serializeToHexString(const T& t) {
   SimpleByteWriter writer;
   Serializer serializer(writer);
   serializer.writeProperty(0, t);
   std::string hexString;
   for (size_t i = 0; i < writer.size(); i++) {
      hexString += "0123456789ABCDEF"[static_cast<uint8_t>(writer.data()[i]) >> 4];
      hexString += "0123456789ABCDEF"[static_cast<uint8_t>(writer.data()[i]) & 0xF];
   }
   return hexString;
}

} //end namespace lingodb::utility
#endif //LINGODB_UTILITY_SERIALIZATION_H
