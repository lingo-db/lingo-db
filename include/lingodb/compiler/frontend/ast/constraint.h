#pragma once
#include <cstdint>
#include <lingodb/compiler/frontend/sql-parser/gen/location.hh>
#include <vector>
namespace lingodb::ast {
enum class ConstraintType : std::uint8_t {
   INVALID = 0,     // invalid constraint type
   NOT_NULL = 1,    // NOT NULL constraint
   CHECK = 2,       // CHECK constraint
   UNIQUE = 3,      // UNIQUE constraint
   FOREIGN_KEY = 4, // FOREIGN KEY constraint
   NULLABLE = 5, // NULLABLE constraint
};

class Constraint {
   public:
   Constraint(ConstraintType type) : type(type){}

   ConstraintType type; // The type of the constraint
   location loc;


};

class UniqueConstraint : public Constraint {
   public:
   explicit UniqueConstraint(bool isPrimaryKey) : Constraint(ConstraintType::UNIQUE), columnNames({}), isPrimaryKey(isPrimaryKey) {}
   UniqueConstraint( std::vector<std::string> columnNames, bool isPrimaryKey) : Constraint(ConstraintType::UNIQUE), columnNames(columnNames), isPrimaryKey(isPrimaryKey) {}

   //! Whether this unique constraint is a primary key
   bool isPrimaryKey = false;

   //! The names of the columns that are part of the unique constraint. Empty if the constraint is directly written next to the columnDef: e.g. CREATE TABLE t (a INT UNIQUE)
   std::vector<std::string> columnNames;
};

}