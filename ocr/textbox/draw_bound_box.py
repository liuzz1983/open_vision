        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bounding_boxes)
    
        # Draw the bounding box in an image summary.
        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                      bbox_for_draw)
        tf.image_summary('images_with_box', image_with_box)
    
        # Employ the bounding box to distort the image.
        distorted_image = tf.slice(image, begin, size)