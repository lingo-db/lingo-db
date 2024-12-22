#ifndef LINGODB_UTILITY_SERIALIZATION_H
#define LINGODB_UTILITY_SERIALIZATION_H
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>
namespace lingodb::utility {
struct ByteWriter {
   virtual void write(const std::byte* data, size_t size) = 0;
   template <class T>
   void write(const T& t) {
      static_assert(std::is_standard_layout<T>::value, "T must be standard layout type");
      write(reinterpret_cast<const std::byte*>(&t), sizeof(T));
   }
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
class Serializer {
   ByteWriter& writer;

   public:
   Serializer(ByteWriter& writer) : writer(writer) {}

   private:
   template <std::same_as<bool> T>
   void writeValue(T value) {
      writer.write(value);
   }
   template <std::same_as<uint16_t> T>
   void writeValue(T value) {
      writer.write(value);
   }
   template <std::same_as<int> T>
   void writeValue(T value) {
      writer.write(value);
   }
   template <std::same_as<float> T>
   void writeValue(T value) {
      writer.write(value);
   }
   template <std::same_as<double> T>
   void writeValue(T value) {
      writer.write(value);
   }
   template <std::same_as<size_t> T>
   void writeValue(T value) {
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
   template <std::same_as<std::string> T>
   T read() {
      size_t length = read<size_t>();
      std::string str;
      str.resize(length);
      reader.read(reinterpret_cast<std::byte*>(str.data()), length);
      return str;
   }
   template <IsUniquePtr T>
   T read() {
      if (read<marker_t>() == notPresent) {
         return nullptr;
      }
      return std::make_unique<typename T::element_type>(read<typename T::element_type>());
   }
   template <IsSharedPtr T>
   T read() {
      if (read<marker_t>() == notPresent) {
         return nullptr;
      }
      return std::make_shared<typename T::element_type>(read<typename T::element_type>());
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
   void startObject() {
      if (read<marker_t>() != objectStart) {
         throw std::runtime_error("Expected object start marker");
      }
   }
   void endObject() {
      if (read<marker_t>() != objectEnd) {
         throw std::runtime_error("Expected object end marker");
      }
   }
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
      auto startMarker = read<marker_t>();
      if (startMarker != propertyId) {
         throw std::runtime_error("Expected property start marker");
      }
      T t = read<T>();
      if (read<marker_t>() != propertyId) {
         throw std::runtime_error("Expected property end marker");
      }
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
   void read(std::byte* data, size_t size) override {
      if (size > this->size) {
         throw std::runtime_error("Read past end of buffer");
      }
      std::copy(buffer, buffer + size, data);
      buffer += size;
      this->size -= size;
   }
};

} //end namespace lingodb::utility
#endif //LINGODB_UTILITY_SERIALIZATION_H
