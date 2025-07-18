set persist=1;
CREATE TABLE customer (
                          c_customer_sk             BIGINT NOT NULL,
                          c_customer_id             CHAR(16) NOT NULL,
                          c_current_cdemo_sk        BIGINT,
                          c_current_hdemo_sk        BIGINT,
                          c_current_addr_sk         BIGINT,
                          c_first_shipto_date_sk    BIGINT,
                          c_first_sales_date_sk     BIGINT,
                          c_salutation              STRING,
                          c_first_name              CHAR(20),
                          c_last_name               CHAR(30),
                          c_preferred_cust_flag     CHAR(1),
                          c_birth_day               INT,
                          c_birth_month             INT,
                          c_birth_year              INT,
                          c_birth_country           VARCHAR(20),
                          c_login                   CHAR(13),
                          c_email_address           CHAR(50),
                          c_last_review_date        STRING
);
CREATE TABLE customer_address (
                                  ca_address_sk             BIGINT NOT NULL,
                                  ca_address_id             STRING NOT NULL,
                                  ca_street_number          STRING,
                                  ca_street_name            STRING,
                                  ca_street_type            STRING,
                                  ca_suite_number           STRING,
                                  ca_city                   STRING,
                                  ca_county                 VARCHAR(20),
                                  ca_state                  CHAR(2),
                                  ca_zip                    STRING,
                                  ca_country                STRING,
                                  ca_gmt_offset             DECIMAL(5,2),
                                  ca_location_type          STRING
);
CREATE TABLE customer_demographics (
                                       cd_demo_sk                BIGINT NOT NULL,
                                       cd_gender                 CHAR(1),
                                       cd_marital_status         CHAR(1),
                                       cd_education_status       CHAR(20),
                                       cd_purchase_estimate      INT,
                                       cd_credit_rating          STRING,
                                       cd_dep_count              INT,
                                       cd_dep_employed_count     INT,
                                       cd_dep_college_count      INT
);
CREATE TABLE date_dim (
                          d_date_sk                 BIGINT NOT NULL,
                          d_date_id                 STRING NOT NULL,
                          d_date                    date,
                          d_month_seq               INT,
                          d_week_seq                INT,
                          d_quarter_seq             INT,
                          d_year                    INT,
                          d_dow                     INT,
                          d_moy                     INT,
                          d_dom                     INT,
                          d_qoy                     INT,
                          d_fy_year                 INT,
                          d_fy_quarter_seq          INT,
                          d_fy_week_seq             INT,
                          d_day_name                STRING,
                          d_quarter_name            STRING,
                          d_holiday                 STRING,
                          d_weekend                 STRING,
                          d_following_holiday       STRING,
                          d_first_dom               INT,
                          d_last_dom                INT,
                          d_same_day_ly             INT,
                          d_same_day_lq             INT,
                          d_current_day             STRING,
                          d_current_week            STRING,
                          d_current_month           STRING,
                          d_current_quarter         STRING,
                          d_current_year            STRING
);
CREATE TABLE household_demographics (
                                        hd_demo_sk                BIGINT NOT NULL,
                                        hd_income_band_sk         BIGINT,
                                        hd_buy_potential          STRING,
                                        hd_dep_count              INT,
                                        hd_vehicle_count          INT
);
CREATE TABLE income_band (
                             ib_income_band_sk         BIGINT NOT NULL,
                             ib_lower_bound            INT,
                             ib_upper_bound            INT
);
CREATE TABLE inventory (
                           inv_date_sk               BIGINT NOT NULL,
                           inv_item_sk               BIGINT NOT NULL,
                           inv_warehouse_sk          BIGINT NOT NULL,
                           inv_quantity_on_hand      INT
);
CREATE TABLE item (
                      i_item_sk                 BIGINT NOT NULL,
                      i_item_id                 CHAR(16) NOT NULL,
                      i_rec_start_date          STRING,
                      i_rec_end_date            STRING,
                      i_item_desc               VARCHAR(200),
                      i_current_price           DECIMAL(7,2),
                      i_wholesale_cost          DECIMAL(7,2),
                      i_brand_id                INT,
                      i_brand                   STRING,
                      i_class_id                INT,
                      i_class                   STRING,
                      i_category_id             INT,
                      i_category                CHAR(50),
                      i_manufact_id             INT,
                      i_manufact                STRING,
                      i_size                    STRING,
                      i_formulation             STRING,
                      i_color                   STRING,
                      i_units                   STRING,
                      i_container               STRING,
                      i_manager_id              INT,
                      i_product_name            STRING
);
CREATE TABLE item_marketprices (
                                   imp_sk                  BIGINT NOT NULL,
                                   imp_item_sk             BIGINT NOT NULL,
                                   imp_competitor          STRING,
                                   imp_competitor_price    DECIMAL(7,2),
                                   imp_start_date          BIGINT,
                                   imp_end_date            BIGINT
);
CREATE TABLE product_reviews (
                                pr_review_sk            BIGINT NOT NULL,
                                pr_review_date          date,
                                pr_review_time          STRING,
                                pr_review_rating        INT NOT NULL,
                                pr_item_sk              BIGINT NOT NULL,
                                pr_user_sk              BIGINT,
                                pr_order_sk             BIGINT,
                                pr_review_content       STRING NOT NULL
);
CREATE TABLE promotion (
                           p_promo_sk               BIGINT NOT NULL,
                           p_promo_id               STRING NOT NULL,
                           p_start_date_sk          BIGINT,
                           p_end_date_sk            BIGINT,
                           p_item_sk                BIGINT,
                           p_cost                   DECIMAL(15,2),
                           p_response_target        INT,
                           p_promo_name             STRING,
                           p_channel_dmail          CHAR(1),
                           p_channel_email          CHAR(1),
                           p_channel_catalog        STRING,
                           p_channel_tv             CHAR(1),
                           p_channel_radio          STRING,
                           p_channel_press          STRING,
                           p_channel_event          STRING,
                           p_channel_demo           STRING,
                           p_channel_details        STRING,
                           p_purpose                STRING,
                           p_discount_active        STRING
);
CREATE TABLE reason (
                        r_reason_sk             BIGINT NOT NULL,
                        r_reason_id             STRING NOT NULL,
                        r_reason_desc           STRING
);
CREATE TABLE ship_mode (
                           sm_ship_mode_sk          BIGINT NOT NULL,
                           sm_ship_mode_id          STRING NOT NULL,
                           sm_type                  STRING,
                           sm_code                  STRING,
                           sm_carrier               STRING,
                           sm_contract              STRING
);
CREATE TABLE store (
                       s_store_sk              BIGINT NOT NULL,
                       s_store_id              CHAR(16) NOT NULL,
                       s_rec_start_date        STRING,
                       s_rec_end_date          STRING,
                       s_closed_date_sk        BIGINT,
                       s_store_name            VARCHAR(50),
                       s_number_employees      INT,
                       s_floor_space           INT,
                       s_hours                 STRING,
                       s_manager               STRING,
                       s_market_id             INT,
                       s_geography_class       STRING,
                       s_market_desc           STRING,
                       s_market_manager        STRING,
                       s_division_id           INT,
                       s_division_name         STRING,
                       s_company_id            INT,
                       s_company_name          STRING,
                       s_street_number         STRING,
                       s_street_name           STRING,
                       s_street_type           STRING,
                       s_suite_number          STRING,
                       s_city                  STRING,
                       s_county                STRING,
                       s_state                 STRING,
                       s_zip                   STRING,
                       s_country               STRING,
                       s_gmt_offset            DECIMAL(5,2),
                       s_tax_precentage        DECIMAL(5,2)
);
CREATE TABLE store_returns (
                               sr_returned_date_sk       BIGINT, -- default 9999999,
                               sr_return_time_sk         BIGINT,
                               sr_item_sk                BIGINT NOT NULL,
                               sr_customer_sk            BIGINT,
                               sr_cdemo_sk               BIGINT,
                               sr_hdemo_sk               BIGINT,
                               sr_addr_sk                BIGINT,
                               sr_store_sk               BIGINT,
                               sr_reason_sk              BIGINT,
                               sr_ticket_number          BIGINT NOT NULL,
                               sr_return_quantity        INT,
                               sr_return_amt             DECIMAL(7,2),
                               sr_return_tax             DECIMAL(7,2),
                               sr_return_amt_inc_tax     DECIMAL(7,2),
                               sr_fee                    DECIMAL(7,2),
                               sr_return_ship_cost       DECIMAL(7,2),
                               sr_refunded_cash          DECIMAL(7,2),
                               sr_reversed_charge        DECIMAL(7,2),
                               sr_store_credit           DECIMAL(7,2),
                               sr_net_loss               DECIMAL(7,2)
);
CREATE TABLE store_sales (
                             ss_sold_date_sk           BIGINT, -- DEFAULT 9999999,
                             ss_sold_time_sk           BIGINT,
                             ss_item_sk                BIGINT NOT NULL,
                             ss_customer_sk            BIGINT,
                             ss_cdemo_sk               BIGINT,
                             ss_hdemo_sk               BIGINT,
                             ss_addr_sk                BIGINT,
                             ss_store_sk               BIGINT,
                             ss_promo_sk               BIGINT,
                             ss_ticket_number          BIGINT NOT NULL,
                             ss_quantity               INT,
                             ss_wholesale_cost         DECIMAL(7,2),
                             ss_list_price             DECIMAL(7,2),
                             ss_sales_price            DECIMAL(7,2),
                             ss_ext_discount_amt       DECIMAL(7,2),
                             ss_ext_sales_price        DECIMAL(7,2),
                             ss_ext_wholesale_cost     DECIMAL(7,2),
                             ss_ext_list_price         DECIMAL(7,2),
                             ss_ext_tax                DECIMAL(7,2),
                             ss_coupon_amt             DECIMAL(7,2),
                             ss_net_paid               DECIMAL(7,2),
                             ss_net_paid_inc_tax       DECIMAL(7,2),
                             ss_net_profit             DECIMAL(7,2)
);
CREATE TABLE time_dim (
                          t_time_sk                BIGINT NOT NULL,
                          t_time_id                STRING NOT NULL,
                          t_time                   INT,
                          t_hour                   INT,
                          t_minute                 INT,
                          t_second                 INT,
                          t_am_pm                  STRING,
                          t_shift                  STRING,
                          t_sub_shift              STRING,
                          t_meal_time              STRING
);
CREATE TABLE warehouse (
                           w_warehouse_sk           BIGINT NOT NULL,
                           w_warehouse_id           STRING NOT NULL,
                           w_warehouse_name         VARCHAR(20),
                           w_warehouse_sq_ft        INT,
                           w_street_number          STRING,
                           w_street_name            STRING,
                           w_street_type            STRING,
                           w_suite_number           STRING,
                           w_city                   STRING,
                           w_county                 STRING,
                           w_state                  CHAR(2),
                           w_zip                    STRING,
                           w_country                STRING,
                           w_gmt_offset             DECIMAL(5,2)
);
CREATE TABLE web_clickstreams (
                                  wcs_click_date_sk       BIGINT,
                                  wcs_click_time_sk       BIGINT,
                                  wcs_sales_sk            BIGINT,
                                  wcs_item_sk             BIGINT,
                                  wcs_web_page_sk         BIGINT,
                                  wcs_user_sk             BIGINT
);
CREATE TABLE web_page (
                          wp_web_page_sk            BIGINT NOT NULL,
                          wp_web_page_id            STRING NOT NULL,
                          wp_rec_start_date         STRING,
                          wp_rec_end_date           STRING,
                          wp_creation_date_sk       BIGINT,
                          wp_access_date_sk         BIGINT,
                          wp_autogen_flag           STRING,
                          wp_customer_sk            BIGINT,
                          wp_url                    STRING,
                          wp_type                   CHAR(50),
                          wp_char_count             INT,
                          wp_link_count             INT,
                          wp_image_count            INT,
                          wp_max_ad_count           INT
);
CREATE TABLE web_returns (
                             wr_returned_date_sk        BIGINT, -- default 9999999,
                             wr_returned_time_sk        BIGINT,
                             wr_item_sk                 BIGINT NOT NULL,
                             wr_refunded_customer_sk    BIGINT,
                             wr_refunded_cdemo_sk       BIGINT,
                             wr_refunded_hdemo_sk       BIGINT,
                             wr_refunded_addr_sk        BIGINT,
                             wr_returning_customer_sk   BIGINT,
                             wr_returning_cdemo_sk      BIGINT,
                             wr_returning_hdemo_sk      BIGINT,
                             wr_returning_addr_sk       BIGINT,
                             wr_web_page_sk             BIGINT,
                             wr_reason_sk               BIGINT,
                             wr_order_number            BIGINT NOT NULL,
                             wr_return_quantity         INT,
                             wr_return_amt              DECIMAL(7,2),
                             wr_return_tax              DECIMAL(7,2),
                             wr_return_amt_inc_tax      DECIMAL(7,2),
                             wr_fee                     DECIMAL(7,2),
                             wr_return_ship_cost        DECIMAL(7,2),
                             wr_refunded_cash           DECIMAL(7,2),
                             wr_reversed_charge         DECIMAL(7,2),
                             wr_account_credit          DECIMAL(7,2),
                             wr_net_loss                DECIMAL(7,2)
);
CREATE TABLE web_sales (
                           ws_sold_date_sk            BIGINT,
                           ws_sold_time_sk            BIGINT,
                           ws_ship_date_sk            BIGINT, -- default 9999999,
                           ws_item_sk                 BIGINT NOT NULL,
                           ws_bill_customer_sk        BIGINT,
                           ws_bill_cdemo_sk           BIGINT,
                           ws_bill_hdemo_sk           BIGINT,
                           ws_bill_addr_sk            BIGINT,
                           ws_ship_customer_sk        BIGINT,
                           ws_ship_cdemo_sk           BIGINT,
                           ws_ship_hdemo_sk           BIGINT,
                           ws_ship_addr_sk            BIGINT,
                           ws_web_page_sk             BIGINT,
                           ws_web_site_sk             BIGINT,
                           ws_ship_mode_sk            BIGINT,
                           ws_warehouse_sk            BIGINT,
                           ws_promo_sk                BIGINT,
                           ws_order_number            BIGINT NOT NULL,
                           ws_quantity                INT,
                           ws_wholesale_cost          DECIMAL(7,2),
                           ws_list_price              DECIMAL(7,2),
                           ws_sales_price             DECIMAL(7,2),
                           ws_ext_discount_amt        DECIMAL(7,2),
                           ws_ext_sales_price         DECIMAL(7,2),
                           ws_ext_wholesale_cost      DECIMAL(7,2),
                           ws_ext_list_price          DECIMAL(7,2),
                           ws_ext_tax                 DECIMAL(7,2),
                           ws_coupon_amt              DECIMAL(7,2),
                           ws_ext_ship_cost           DECIMAL(7,2),
                           ws_net_paid                DECIMAL(7,2),
                           ws_net_paid_inc_tax        DECIMAL(7,2),
                           ws_net_paid_inc_ship       DECIMAL(7,2),
                           ws_net_paid_inc_ship_tax   DECIMAL(7,2),
                           ws_net_profit              DECIMAL(7,2)
);
CREATE TABLE web_site (
                          web_site_sk               BIGINT NOT NULL,
                          web_site_id               STRING NOT NULL,
                          web_rec_start_date        STRING,
                          web_rec_end_date          STRING,
                          web_name                  STRING,
                          web_open_date_sk          BIGINT,
                          web_close_date_sk         BIGINT,
                          web_class                 STRING,
                          web_manager               STRING,
                          web_mkt_id                INT,
                          web_mkt_class             STRING,
                          web_mkt_desc              STRING,
                          web_market_manager        STRING,
                          web_company_id            INT,
                          web_company_name          STRING,
                          web_street_number         STRING,
                          web_street_name           STRING,
                          web_street_type           STRING,
                          web_suite_number          STRING,
                          web_city                  STRING,
                          web_county                STRING,
                          web_state                 STRING,
                          web_zip                   STRING,
                          web_country               STRING,
                          web_gmt_offset            DECIMAL(5,2),
                          web_tax_percentage        DECIMAL(5,2)
);


COPY customer FROM 'customer.dat' DELIMITER '|';
COPY customer_address FROM 'customer_address.dat' DELIMITER '|';
COPY customer_demographics FROM 'customer_demographics.dat' DELIMITER '|';
COPY date_dim FROM 'date_dim.dat' DELIMITER '|';
COPY household_demographics FROM 'household_demographics.dat' DELIMITER '|';
COPY income_band FROM 'income_band.dat' DELIMITER '|';
COPY inventory FROM 'inventory.dat' DELIMITER '|';
COPY item FROM 'item.dat' DELIMITER '|';
COPY item_marketprices FROM 'item_marketprices.dat' DELIMITER '|';
COPY product_reviews FROM 'product_reviews.dat' DELIMITER '|';
COPY promotion FROM 'promotion.dat' DELIMITER '|';
COPY reason FROM 'reason.dat' DELIMITER '|';
COPY ship_mode FROM 'ship_mode.dat' DELIMITER '|';
COPY store FROM 'store.dat' DELIMITER '|';
COPY store_returns FROM 'store_returns.dat' DELIMITER '|';
COPY store_sales FROM 'store_sales.dat' DELIMITER '|';
COPY time_dim FROM 'time_dim.dat' DELIMITER '|';
COPY warehouse FROM 'warehouse.dat' DELIMITER '|';
COPY web_clickstreams FROM 'web_clickstreams.dat' DELIMITER '|';
COPY web_page FROM 'web_page.dat' DELIMITER '|';
COPY web_returns FROM 'web_returns.dat' DELIMITER '|';
COPY web_sales FROM 'web_sales.dat' DELIMITER '|';
COPY web_site FROM 'web_site.dat' DELIMITER '|';