# Selective spatial constraint graph

This round implements the layout channel separately from anchor proximity and adds a small allowed-edge filter and two-channel score combiner. No graph fusion or weight tuning was run because Gate G failed. The allowed evidence-supported spatial edges remain NEAR, NEXT_TO, ROAD_ASSOCIATION, FOOTPRINT_DISTANCE, LAYOUT_ORDER, ROW_INDEX, COLUMN_INDEX, and ROAD_SEQUENCE. Generic BEHIND, FRONT, and implicit LEFT/RIGHT require an explicit frame; they remain disabled otherwise, as concluded by Gate F.

L4 uses a fixed, untuned 0.5/0.5 sum of layout and distance to a text-bound reference within the parking scope; its results are in the Gate G table. No val_unseen tuning occurred.
