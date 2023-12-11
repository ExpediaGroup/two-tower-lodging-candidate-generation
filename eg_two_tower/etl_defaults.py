"""
Copyright [2023] Expedia, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

DEFAULT_VALIDATION_DATE_SPLIT = "2021-07-24"
IMPRESSION_COLUMNS = [("rank", "integer"), ("prop_id", "long"), ("is_travel_ad", "integer"),
                      ("review_rating", "integer"),
                      ("review_count", "integer"), ("star_rating", "float"), ("is_free_cancellation", "integer"),
                      ("is_drr", "integer"), ("price_bucket", "integer"), ("num_clicks", "integer"),
                      ("is_trans", "integer")]
FINAL_QUERY_COLUMNS = [
    'user_id',
    'search_timestamp',
    'point_of_sale',
    'geo_location_country',
    'is_mobile',
    'destination_id',
    'adult_count',
    'child_count',
    'infant_count',
    'room_count',
    'sort_type',
    "applied_filters"
]
FINAL_IMPRESSION_COLUMNS = [
    'impression_prop_id',
    'impression_review_rating',
    'impression_review_count',
    'impression_star_rating',
    'impression_price_bucket',
    'impression_num_clicks',
    'impression_is_trans'
]
FINAL_AMENITY_COLUMNS = [
    'AirConditioning',
    'AirportTransfer',
    'Bar',
    'FreeAirportTransportation',
    'FreeBreakfast',
    'FreeParking',
    'FreeWiFi',
    'Gym',
    'HighSpeedInternet',
    'HotTub',
    'LaundryFacility',
    'Parking',
    'PetsAllowed',
    'PrivatePool',
    'SpaServices',
    'SwimmingPool',
    'WasherDryer',
    'WiFi'
]

FINAL_DERIVED_FEATURE_COLUMNS = [
    'clicked',
    'search_timestamp_month',
    'search_timestamp_dayofweek',
    'checkin_date_month',
    'checkin_date_dayofweek',
    'checkout_date_month',
    'checkout_date_dayofweek',
    'days_till_trip',
    'length_of_stay'
]
CANDIDATE_IMPRESSION_COLUMNS = [
    'impression_review_rating',
    'impression_review_count',
    'impression_star_rating',
    'impression_price_bucket'
]
QUERY_CATEGORICAL_COLUMNS = [
    'user_id',
    'point_of_sale',
    'geo_location_country',
    'destination_id',
    'sort_type',
    'search_timestamp_month',
    'search_timestamp_dayofweek',
    'checkin_date_month',
    'checkin_date_dayofweek',
    'checkout_date_month',
    'checkout_date_dayofweek',
    "applied_filters"
]

IMPRESSION_CATEGORICAL_COLUMNS = [
    'impression_prop_id',
    'impression_review_rating',
    'impression_review_count',
    'impression_star_rating',
    'impression_price_bucket'
]

NUMERICAL_COLUMNS = [
    'days_till_trip',
    'length_of_stay',
    'is_mobile',
    'adult_count',
    'child_count',
    'infant_count',
    'room_count',
    'AirConditioning',
    'AirportTransfer',
    'Bar',
    'FreeAirportTransportation',
    'FreeBreakfast',
    'FreeParking',
    'FreeWiFi',
    'Gym',
    'HighSpeedInternet',
    'HotTub',
    'LaundryFacility',
    'Parking',
    'PetsAllowed',
    'PrivatePool',
    'SpaServices',
    'SwimmingPool',
    'WasherDryer',
    'WiFi'
]
FINAL_COLUMNS = FINAL_QUERY_COLUMNS + FINAL_DERIVED_FEATURE_COLUMNS + FINAL_IMPRESSION_COLUMNS + FINAL_AMENITY_COLUMNS
