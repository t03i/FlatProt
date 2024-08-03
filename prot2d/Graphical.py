import svgwrite
import math

def create_arrow_V1(x1,y1,x2,y2, arrow_color="blue", line_width=20):
    dwg = svgwrite.Drawing(profile='full', size=("100%", "100%"))
    #angle for the arrow based on the line coords
    angle = math.atan2(y2 - y1, x2 - x1)
    marker = dwg.marker(insert=(1, 2.5), size=(4, 4))
    marker.add(dwg.polygon(points=[(0, 1), (1,2),(1,3), (0, 4)], fill=arrow_color, stroke_width=0.1, stroke="black"))
    
    #marker's orientation to the calculated angle in degrees
    marker['orient'] = "{}deg".format(math.degrees(angle))
    dwg.defs.add(marker)
    line = dwg.add(dwg.polyline(
        [(x1, y1), (x2, y2)],
        stroke=arrow_color, fill='none', stroke_width=line_width))
    outline = dwg.add(dwg.polyline(
        [(x1, y1), (x2, y2)],
        stroke='black', fill='none', stroke_width=line_width+5))
    #set marker
    line.set_markers((None,None, marker))
    dwg.add(outline)
    dwg.add(line)
    return line,dwg

def create_rectangle_between_2_points(x1,y1,x2,y2,color,thickness ):
    dwg = svgwrite.Drawing(profile='full', size=("100%", "100%"))
    dx = x2 - x1
    dy = y2 - y1
    #length = get_vector_length(dx,dy) # calc the euclidian distance between the two points for normalization
    normalized_orthogonal_dx,normalized_orthogonal_dy = get_normalized_orthogonal_vector(dx,dy)
    #normalized_dx,normalized_dy = get_normalized_vector(dx,dy)

    #four vertices of the rectangle
    x1a, y1a = x1 + normalized_orthogonal_dx * thickness / 2, y1 + normalized_orthogonal_dy * thickness / 2
    x1b, y1b = x1 - normalized_orthogonal_dx * thickness / 2, y1 - normalized_orthogonal_dy * thickness / 2
    x2a, y2a = x2 + normalized_orthogonal_dx * thickness / 2, y2 + normalized_orthogonal_dy * thickness / 2
    x2b, y2b = x2 - normalized_orthogonal_dx * thickness / 2, y2 - normalized_orthogonal_dy * thickness / 2
    #polyline with the calculated vertices
    points = [(x1a, y1a), (x1b, y1b), (x2b, y2b), (x2a, y2a),(x1a, y1a)]
    polygon = dwg.polygon(points, fill=color, stroke="black")
    dwg.add(polygon)
    return dwg
    
def create_arrow_line_between_2_points(x1,y1,x2,y2,color,thickness,opacity):
    dwg = svgwrite.Drawing(profile='full', size=("100%", "100%"))
    dx = x2 - x1
    dy = y2 - y1
    normalized_orthogonal_dx,normalized_orthogonal_dy = get_normalized_orthogonal_vector(dx,dy)
    x1a, y1a = x1 + normalized_orthogonal_dx * thickness / 2, y1 + normalized_orthogonal_dy * thickness / 2
    x1b, y1b = x1 - normalized_orthogonal_dx * thickness / 2, y1 - normalized_orthogonal_dy * thickness / 2
    dir_line = dwg.polyline([(x1a, y1a),(x2,y2) ,(x1b, y1b)],stroke='black', fill=color, opacity=opacity)
    return dir_line
def create_helix_between(x1,y1,x2,y2,color,thickness ):
    dwg = svgwrite.Drawing(profile='full', size=("100%", "100%"))
    dx = x2 - x1
    dy = y2 - y1
    #border rectangle:
    #dwg.add(create_rectangle_between_2_points(x1,y1,x2,y2,'none',80))
    normalized_dx,normalized_dy = get_normalized_vector(dx,dy)
    normalized_orthogonal_dx, normalized_orthogonal_dy = get_normalized_orthogonal_vector(dx,dy)
    #helix:
    module_size = thickness/2
    rect_witdh = 2/3*module_size
    next_module_coords= (x1,y1)

    #do as many module as can fit in the space
    remaining_length = get_coord_distance(next_module_coords[0],next_module_coords[1],x2,y2)
    #start element:
    dwg.add(create_helix_start_element(x1,y1,x2,y2,color,thickness,rect_witdh))
    
    while remaining_length >= (module_size+ rect_witdh):
        module, next_module_coords = create_helix_module(next_module_coords[0],next_module_coords[1],x2,y2, color, thickness, rect_witdh)
        dwg.add(module)
        next_module_coords = next_module_coords[0] - (normalized_orthogonal_dx*thickness/2),next_module_coords[1] - (normalized_orthogonal_dy*thickness/2)
        remaining_length = get_coord_distance(next_module_coords[0],next_module_coords[1],x2,y2)

    #end element
    end_element_start=next_module_coords[0] - (normalized_orthogonal_dx*thickness/2),next_module_coords[1] - (normalized_orthogonal_dy*thickness/2)
    end_element_coords = end_element_start[0],end_element_start[1],end_element_start[0] + (normalized_dx*rect_witdh),end_element_start[1] + (normalized_dy*rect_witdh)
    dwg.add(dwg.polygon([(end_element_coords[0],end_element_coords[1]),(end_element_coords[2],end_element_coords[3]),(x2,y2)], fill=color, stroke="black"))
    
    return dwg

def create_helix_start_element(x1,y1,x2,y2,color,thickness,rect_witdh):
    dx = x2 - x1
    dy = y2 - y1 #euclidian distance between the two points for normalization
    normalized_orthogonal_dx,normalized_orthogonal_dy = get_normalized_orthogonal_vector(dx,dy)
    normalized_dx,normalized_dy = get_normalized_vector(dx,dy)
    xr1,yr1 = x1 - (normalized_orthogonal_dx*thickness/2), y1 - (normalized_orthogonal_dy*thickness/2)
    #xr1_spez, yr1_spez = x1 + (normalized_dx*rect_witdh/2), y1 + (normalized_dy*rect_witdh/2)
    xr2,yr2 = xr1 + (normalized_dx*rect_witdh), yr1 + (normalized_dy*rect_witdh)
    dwg = svgwrite.Drawing(profile='full', size=("100%", "100%"))
    start_triangle = dwg.polygon([(x1,y1),(xr1,yr1),(xr2,yr2)], fill=color, stroke="black")
    dwg.add(start_triangle)
    return dwg
def create_helix_module(x1,y1,x2,y2, color, thickness, rect_witdh):
    dwg = svgwrite.Drawing(profile='full', size=("100%", "100%"))
    dx = x2 - x1
    dy = y2 - y1 #euclidian distance between the two points for normalization
    normalized_orthogonal_dx,normalized_orthogonal_dy = get_normalized_orthogonal_vector(dx,dy)
    normalized_dx,normalized_dy = get_normalized_vector(dx,dy)
    
    
    xr1,yr1 = x1 - (normalized_orthogonal_dx*thickness/2), y1 - (normalized_orthogonal_dy*thickness/2)
    #xr1_spez, yr1_spez = x1 + (normalized_dx*rect_witdh/2), y1 + (normalized_dy*rect_witdh/2)
    xr2,yr2 = xr1 + (normalized_dx*rect_witdh), yr1 + (normalized_dy*rect_witdh)
    xr3,yr3 = xr2 + (normalized_orthogonal_dx*thickness), yr2 + (normalized_orthogonal_dy*thickness)
    xr4,yr4 = xr3 + (normalized_dx* rect_witdh), yr3 + (normalized_dy* rect_witdh)
    front_rect=dwg.polygon([(xr1,yr1),(xr2,yr2),(xr4,yr4),(xr3,yr3)], fill=color, stroke="black")
    
    xr5,yr5 = xr4 - (normalized_orthogonal_dx*thickness), yr4 - (normalized_orthogonal_dy*thickness)
    xr6,yr6 = xr5 + (normalized_dx*rect_witdh), yr5 + (normalized_dy*rect_witdh)
    behind_rect = dwg.polygon([(xr3,yr3),(xr4,yr4),(xr6,yr6),(xr5,yr5)], fill=color, stroke="black")
    dwg.add(behind_rect)
    dwg.add(front_rect)
    
    return dwg,(xr4,yr4)
def create_simple_helix_line(x1,y1,x2,y2, color,thickness,rect_widht, cross_width, opacity):
    dwg = svgwrite.Drawing(profile='full', size=("100%", "100%"))
    #dwg.add(create_rectangle_between_2_points(x1,y1,x2,y2,'none',100))
    dx = x2 - x1
    dy = y2 - y1 # calc the euclidian distance between the two points for normalization
    points = get_line_points(x1,y1,x2,y2,thickness,rect_widht,cross_width, 5)
    down_line =  dwg.polyline(points,stroke='black', fill=color, opacity=opacity)
    dwg.add(down_line)
    return dwg
def get_line_points(x1,y1,x2,y2,thickness,rect_widht, cross_width, min_ending_line):
    if x1==x2 and y1==y2:
        return [(x1,y1),(x2,y2)]
    dx,dy= get_dx_dy(x1,y1,x2,y2)
    norm_ort_vec = get_normalized_orthogonal_vector(dx,dy)
    norm_vec = get_normalized_vector(dx,dy)
    points =[]
    #down line:
    points.append(move_point_vec((x1,y1),norm_ort_vec,-rect_widht/2))
    first_down = move_point_double_vec(points[-1],norm_ort_vec, -((thickness/2)-(rect_widht/2)),norm_vec,cross_width/2)
    points.append(first_down)
    dist_reference_point =move_point_vec(first_down,norm_ort_vec,(thickness/2))
    distance = get_coord_distance(dist_reference_point[0],dist_reference_point[1],x2,y2) - min_ending_line

    number_of_segments = distance/cross_width
    for i in range(0,int(number_of_segments)):
        up_point = move_point_double_vec(points[-1],norm_ort_vec,thickness-rect_widht,norm_vec,cross_width/2)
        points.append(up_point)
        #back down:
        down_point = move_point_double_vec(points[-1],norm_ort_vec,-(thickness-rect_widht),norm_vec, cross_width/2)
        points.append(down_point)
    #end point
    low_end_point = move_point_vec((x2,y2),norm_ort_vec, -(rect_widht/2))
    points.append(low_end_point)
    # upper line:
    starting_point = move_point_vec(points[-1], norm_ort_vec,rect_widht)
    points.append(starting_point)
    spez_point = move_point_vec(points[-3], norm_ort_vec,rect_widht)
    points.append(spez_point)
    for i in range(0,int(number_of_segments)):
        up_point = move_point_double_vec(points[-1],norm_ort_vec,thickness-rect_widht,norm_vec,-(cross_width/2))
        points.append(up_point)
        #back down:
        down_point = move_point_double_vec(points[-1], norm_ort_vec,-(thickness-rect_widht), norm_vec, -(cross_width/2))
        points.append(down_point)
    #end point:
    points.append(move_point_vec((x1,y1),norm_ort_vec,rect_widht/2))
    return points

def move_point_vec(start_point,vector,distance):
    if distance == 0:
        return start_point
    final = start_point[0] + (vector[0]*distance),start_point[1] + (vector[1]*distance)
    return final
def move_point_double_vec(start_point,vector1,distance1,vector2,distance2):
    final = start_point[0] + (vector1[0]*distance1) + (vector2[0]*distance2),start_point[1] + (vector1[1]*distance1) + (vector2[1]*distance2)
    return final
def get_dx_dy(x1,y1,x2,y2):
    dx = x2 - x1
    dy = y2 - y1
    return dx,dy
def get_normalized_vector(dx,dy):
    length = get_vector_length(dx,dy)
    normalized_dx = dx/length
    normalized_dy = dy/length
    return normalized_dx,normalized_dy
def get_normalized_orthogonal_vector(dx, dy):
    length = get_vector_length(dx,dy)
    normalized_orthogonal_dx = -dy / length
    normalized_orthogonal_dy = dx / length
    return normalized_orthogonal_dx, normalized_orthogonal_dy
def get_vector_length(dx, dy):
    length = (dx ** 2 + dy ** 2) ** 0.5
    return length
def get_coord_distance(x1,y1,x2,y2):
    dx = x2 - x1
    dy = y2 - y1
    return get_vector_length(dx,dy)
def testing():

    x1, y1 = 124,20
    x2, y2 = 200,200
    dwg = svgwrite.Drawing('connecting_helix_test.svg', profile='full', width='100%', height='100%')
    dwg.add(svgwrite.Drawing().circle(center=(x1, y1), r=1, fill='red', stroke='none'))
    dwg.add(svgwrite.Drawing().circle(center=(x2, y2), r=1, fill='blue', stroke='none'))
    #
    #dwg.add(create_rectangle_between_2_points(x1,y1,x2,y2, 'green',80))
    #dwg.add(create_arrow_line_between_2_points(x1,y1,x2,y2, 'none',80))
    dwg.add(create_helix_between(x1,y1,x2,y2, 'red', 80))
    dwg.save()
    #print("Done!")

def do_lines_intersect(line1, line2):
    return line1.intersects(line2)

def closest_point_on_line(point, line):
    x, y = point
    x1, y1 = line[0]
    x2, y2 = line[1]
    #direction vector of the line
    dx = x2 - x1
    dy = y2 - y1
    #squared length of the line segment
    line_length_squared = dx**2 + dy**2
    #vector from the starting point of the line to the given point
    delta_x = x - x1
    delta_y = y - y1
    #dot product of the line vector and the vector to the point
    dot_product = (delta_x * dx + delta_y * dy) / line_length_squared
    #coordinates of the closest point on the line
    #point has to be on line segment
    t = max(0, min(1, dot_product))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return closest_x, closest_y