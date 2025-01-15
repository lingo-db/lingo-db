* Storage *Interface*: Apache Arrow with e.g., a little bit of cusomization
* Data Interface for compiled code: custom vector format
  * After pushed-down filters (densely packed)
  * After simplifying transformations:
    * e.g., dictionary decoding
    * arrow string -> custom string format
    * timing stuff -> normalize to nanoseconds since ...
    * if talking about nullable data -> fixed-sized vector with all ones
  * Reasoning: we can still process arbitrary Apache Arrow data (can use Apache Arrow readers etc)
  * But: we do not need to talk Apache Arrow in the compiled/execution part
* Allocation classes:
  * "Morsel": available only during processing of current work unit
    * Also implies: no runtime information about storage available
  * "Managed Lifetime": object/memory is managed by LingoDB runtime
    * Before the object, we store a byte B with the rough management/lifetime type
      * e.g., lives for the whole query execution (a constant)
      * e.g., is reference counted (counter is before B)
      * ...
  * 