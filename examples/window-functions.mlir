%1 = relalg.window %rel, orderby: [(@attr3,desc),(@attr4,asc)] partitionby: [@attr1,@attr2] (%w : !relalg.window, %tuple : !relalg.tuple){
    %2 = relalg.get_current_row %w : !db.int
    %3 = relalg.window_size %w : !db.int
    %4 = db.constant(2) :!db.int
    %5 = db.div %3, $4
    %6 = relalg.get_tuple_from_window %w %5
    %7 = relalg.subwindow_by_rows %w %5,%2 :!relalg.window 
    %8 = relalg.subwindow_by_range %w ...
    %8 = relalg.window_as_relation %7 :!relalg.relation
    %9 = relalg.aggrfn sum min @partsupp::@ps_supplycost %8 : !db.decimal<15,2,nullable>

}