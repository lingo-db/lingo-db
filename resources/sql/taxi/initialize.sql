SET persist=1;
create table taxi_rides(
    vendor_id	integer NOT NULL,
    p_t	timestamp NOT NULL,
    d_t	timestamp NOT NULL,
    passenger_count	integer NOT NULL,
    trip_distance	float NOT NULL,
    p_lon	float NOT NULL,
    p_lat	float NOT NULL,
    ratecode_id	integer NOT NULL,
    store_and_fwd_flag	text NOT NULL,
    d_lon	float NOT NULL,
    d_lat	float NOT NULL,
    payment_type	integer NOT NULL,
    fare_amount	decimal(9,2) NOT NULL,
    extra	decimal(9,2) NOT NULL,
    mta_tax	decimal(9,2) NOT NULL,
    tip_amount	decimal(9,2) NOT NULL,
    tolls_amount	decimal(9,2) NOT NULL,
    improvement_surcharge	decimal(9,2) NOT NULL,
    total_amount	decimal(9,2) NOT NULL
);
copy taxi_rides from 'yellow_tripdata_2016-01.csv' csv;
