#include "lingodb/utility/Serialization.h"

void lingodb::utility::Deserializer::startObject() {
   if (read<marker_t>() != objectStart) {
      throw std::runtime_error("Expected object start marker");
   }
}
void lingodb::utility::Deserializer::endObject() {
   if (read<marker_t>() != objectEnd) {
      throw std::runtime_error("Expected object end marker");
   }
}

void lingodb::utility::SimpleByteReader::read(std::byte* data, size_t size) {
   if (size > this->size) {
      throw std::runtime_error("Read past end of buffer");
   }
   std::copy(buffer, buffer + size, data);
   buffer += size;
   this->size -= size;
}

void lingodb::utility::Deserializer::startProperty(marker_t propertyId) {
   auto startMarker = read<marker_t>();

   if (startMarker != propertyId) {
      throw std::runtime_error("Expected property start marker");
   }
}
void lingodb::utility::Deserializer::endProperty(marker_t propertyId) {
   if (read<marker_t>() != propertyId) {
      throw std::runtime_error("Expected property end marker");
   }
}

lingodb::utility::FileByteReader::FileByteReader(std::string filename) : istream(filename, std::ios::binary) {
   if (!istream) {
      throw std::runtime_error("Failed to open file for reading");
   }
}
void lingodb::utility::FileByteReader::read(std::byte* data, size_t size) {
   istream.read(reinterpret_cast<char*>(data), size);
   if (!istream) {
      throw std::runtime_error("Failed to read from file");
   }
}
lingodb::utility::FileByteWriter::FileByteWriter(std::string filename) {
   ostream.open(filename, std::ios::binary);
   if (!ostream) {
      throw std::runtime_error("Failed to open file for writing");
   }
}
void lingodb::utility::FileByteWriter::write(const std::byte* data, size_t size) {
   ostream.write(reinterpret_cast<const char*>(data), size);
   if (!ostream) {
      throw std::runtime_error("Failed to write to file");
   }
}
