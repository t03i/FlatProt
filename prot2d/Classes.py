import svgwrite
from Bio.PDB import DSSP
from Bio import PDB
import re
import numpy as np
from itertools import product,combinations
from shapely.geometry import LineString
import math
from tqdm import tqdm
import random
import matplotlib.colors as mcolors
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import copy
from .Graphical import*



class cross_class_functions():
    def calc_summarising_point(residue_segment):
        coordinate_segment = [(residue.x,residue.y) for residue in residue_segment]
        avg_coord = cross_class_functions.combine_coords(coordinate_segment)
        sum_point = Summarised_point(residue_segment, avg_coord)
        [residue.set_included_in(sum_point) for residue in residue_segment]
        return sum_point
    
    def combine_coords(coords_array):
        averages_tuple = tuple(sum(x) / len(coords_array) for x in zip(*coords_array))
        return averages_tuple
    
    def divide_list_into_parts(input_list, num_parts):
        part_size = (len(input_list) + num_parts - 1) // num_parts
        divided_list = [input_list[i * (part_size - 1):i * (part_size - 1) + part_size] for i in range(num_parts)]
        return divided_list
    
    def calc_ss_percentages(residue_list):
        helix_num = 0
        sheet_num = 0
        coil_num = 0
        for res in residue_list:
            if res.ss == 'helix':
                helix_num+=1
            if res.ss == 'sheet':
                sheet_num+=1
            if res.ss == 'coil':
                coil_num+=1
        return [helix_num/len(residue_list),sheet_num/len(residue_list),coil_num/len(residue_list)]

    def get_DSSP_SS(prot_structure, pdb_file):
        dssp = DSSP(prot_structure[0], pdb_file)
        #Access secondary prot_structure information
        res_ss = {}
        for dssp_data in dssp:
            res_id = dssp_data[0]  # Residue ID
            res_as = dssp_data[1]
            ss = dssp_data[2]  # Secondary prot_structure (H: Helix, E: Strand, C: Coil)
            res_ss[res_id] = ss
        if len(res_ss) == 0:
            raise Exception("DSSP failed to prduce SS annotation! Maybe HEADER missing in pdb-file (selfmade Error)")
        return res_ss

    def simplify_DSSP_output(secondary_structure_string):
        # convert 8 types to 3
        three_type_ss_string = secondary_structure_string.replace('G','H') #Helix-3 to Helix
        three_type_ss_string = three_type_ss_string.replace('I','H') #Helix-5 to Helix
        three_type_ss_string = three_type_ss_string.replace('B','E') #Beta_bridge to Strand
        return three_type_ss_string

    def get_ss_blocks(residue_ss, residues):
        ss_string = cross_class_functions.simplify_DSSP_output(''.join(residue_ss.values()))
        print("SS_String:"+ss_string)
        # find blcoks of at least 4 consecutive Hs
        helix_blocks = [(match.start(), match.end() - 1) for match in re.finditer(r'H{4,}', ss_string)]
        #print(helix_blocks)
        # find blcoks of at least 3 consecutive E or Bs
        sheet_blocks = [(match.start(), match.end() - 1) for match in re.finditer(r'[E]{3,}', ss_string)]
        #print(sheet_blocks)
        coil_string = ""
        for i, char in enumerate(ss_string):
            # Check if the current position is part of any block
            is_in_block = any(start <= i <= end for start, end in helix_blocks) or any(start <= i <= end for start, end in sheet_blocks)
            # Replace the character with "$" if it's in a block, otherwise keep the original character
            coil_string += char if is_in_block else '$'
        coil_blocks = [(match.start(), match.end() - 1) for match in re.finditer(r'[$]{1,}', coil_string)]
        
        helix_blocks = [(start + 1, end + 1) for start, end in helix_blocks]
        sheet_blocks = [(start + 1, end + 1) for start, end in sheet_blocks]
        coil_blocks = [(start + 1, end + 1) for start, end in coil_blocks]

        all_blocks = helix_blocks+sheet_blocks+coil_blocks
        def sort_by_first_element(tup):
            return tup[0]
        all_corresponding_ss = ['helix'] * len(helix_blocks) + ['sheet'] * len(sheet_blocks) + ['coil'] * len(coil_blocks)
        sorted_lists = sorted(zip(all_blocks, all_corresponding_ss), key=sort_by_first_element)
        all_blocks_sorted, all_corresponding_ss_sorted = zip(*sorted_lists)
        
        # convert block positions in secondary strucutr objects with residudes as attribute
        all_ss_structures_obj = []

        for i, block in enumerate(all_blocks_sorted):
            type = all_corresponding_ss_sorted[i]
            res_list = []
            start = block[0]-1
            end = block[1]-1
            for pos in range(start, end+1):
                res_list.append(residues[pos])
            ss_type = 'coil' if type == 'coil' else type
            all_ss_structures_obj.append(Secondary_Structure(ss_type, res_list))
        return all_ss_structures_obj
    
    def radians_to_rotation_matrix (theta_x_rad, theta_y_rad, theta_z_rad):
        rotation_matrix_x = np.array([[1, 0, 0],
                                    [0, np.cos(theta_x_rad), -np.sin(theta_x_rad)],
                                    [0, np.sin(theta_x_rad), np.cos(theta_x_rad)]])

        rotation_matrix_y = np.array([[np.cos(theta_y_rad), 0, np.sin(theta_y_rad)],
                                    [0, 1, 0],
                                    [-np.sin(theta_y_rad), 0, np.cos(theta_y_rad)]])

        rotation_matrix_z = np.array([[np.cos(theta_z_rad), -np.sin(theta_z_rad), 0],
                                    [np.sin(theta_z_rad), np.cos(theta_z_rad), 0],
                                    [0, 0, 1]])
        
        combined_rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))
        return combined_rotation_matrix
    
    def lighten_color(color, factor=0.4):
        rgb = mcolors.hex2color(color)
        lighter_rgb = [min(1, val + factor) for val in rgb]
        return mcolors.rgb2hex(lighter_rgb)

    def darken_color(color, factor=0.2):
        rgb = mcolors.hex2color(color)
        darker_rgb = [max(0, val - factor) for val in rgb]
        return mcolors.rgb2hex(darker_rgb)

class Residue:

    def __init__(self,amino_acid,chain_pos ):
        self.amino_acid = amino_acid
        self.x = None
        self.y = None
        self.z = None
        self.x_rot = None
        self.y_rot = None
        self.z_rot = None
        self.chain_pos =chain_pos
        self.ss = None
        self.ss_obj = None
        self.included_in = None
        self.atoms = {}
        self.lddt= None
        
    def add_new_model(self,x,y,z):
        self.x_modelwise.append(x)
        self.y_modelwise.append(y)
        self.z_modelwise.append(z)
    
    def equals_residue(self, residue2):
        if residue2==None:
            return False
        if self.amino_acid == residue2.amino_acid and self.chain_pos == residue2.chain_pos:
            return True
        else:
            return False
    
    def set_atoms_average_choords(self):
        for atom in self.atoms.values():
            atom.x = sum(atom.x_modelwise)/len(atom.x_modelwise)
            atom.y = sum(atom.y_modelwise)/len(atom.y_modelwise)
            atom.z = sum(atom.z_modelwise)/len(atom.z_modelwise)
            if atom.name == "CA":
                self.x = atom.x
                self.y = atom.y
                self.z = atom.z
   
    def set_included_in(self, summarised_point):
        if self.included_in !=None:
            print("overriding summarising coord")
        self.included_in=summarised_point
    
    def get_distance_to_residue(self ,residue) -> float:
        x1, y1, z1 = self.x,self.y,self.z
        x2, y2, z2 = residue.x,residue.y,residue.z
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return distance
    
    def get_closest_point(self):
        if self.ss == 'coil':
            if self.included_in != None:
                return self.included_in.x,self.included_in.y
            else:
                return self.x,self.y
        else:
            ss_line = self.ss_obj.get_line()
            #find closest point on that line:
            closest_point = closest_point_on_line((self.x,self.y), ss_line)
            return closest_point

class Secondary_Structure:

    def __init__(self,type,residue_list):
        self.type = type
        self.residue_list = residue_list
        self.start_res = self.residue_list[0]
        self.end_res = self.residue_list[-1]
        self.fru_rein_start = None
        self.fru_rein_end = None
        #self.z = sum(residue.z for residue in self.residue_list) / len(self.residue_list)
        self.z = max(residue.z for residue in self.residue_list)
        self.coil_path = None

    def get_average_path(self, averaging):
        avg_points = []
        if averaging ==1:
            for res in self.residue_list:
                avg_points.append((res.x,res.y))
            return avg_points
        #avg_points.append((self.start_res.x,self.start_res.y))
        for i in range(0, len(self.residue_list), averaging):
            residue_segment = self.residue_list[i:i+averaging]
            sum_point=cross_class_functions.calc_summarising_point(residue_segment)
            avg_points.append((sum_point.x,sum_point.y))
        #avg_points.append((self.end_res.x,self.end_res.y))
        return avg_points
    
    def get_avg_point(self):
        coords=[]
        for res in self.residue_list:
            coords.append((res.x,res.y))
        return cross_class_functions.combine_coords(coords)
    def get_string_residues(self):
        string=""
        for residue in self.residue_list:
            string+=residue.amino_acid+", "
        return string
    def get_middle_point(self):
        mx = (self.start_res.x + self.end_res.x) / 2
        my = (self.start_res.y + self.end_res.y) / 2
        return mx, my
    def get_line(self):
        return [(self.start_res.x,self.start_res.y),(self.end_res.x,self.end_res.y)]
    
class Summarised_point:
    def __init__(self,residue_list, avg_coord):
        self.combined_res_list = residue_list
        self.x,self.y = avg_coord
        self.z =  self.calc_avg_z()
        self.lddt = self.calc_avg_lddt()

    def calc_avg_z(self):
        z_sum= 0
        for res in self.combined_res_list:
            z_sum+= res.z
        return z_sum/len(self.combined_res_list)
    def calc_avg_lddt(self):
        num_lddt = 0
        lddt_sum = 0
        for res in self.combined_res_list:
            if res.lddt != None:
                lddt_sum+= res.lddt
                num_lddt+=1
        avg_lddt = lddt_sum/num_lddt if num_lddt>0 else None
        return avg_lddt

    def get_string_residues(self):
        string=""
        for residue in self.combined_res_list:
            string+=residue.amino_acid+", "
        return string

class Atom:
    def __init__(self,name, first_x,first_y,first_z):
        self.name = name
        self.x_modelwise = [first_x]
        self.y_modelwise = [first_y]
        self.z_modelwise = [first_z]
        self.x= None
        self.y= None
        self.z= None
        
    def append_model_choords(self,x,y,z):
        self.x_modelwise.append(x)
        self.y_modelwise.append(y)
        self.z_modelwise.append(z)
    
    def get_distance_to_atom(self,residue):
        x1, y1, z1 = self.x,self.y,self.z
        x2, y2, z2 = residue.x,residue.y,residue.z
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return distance
    
    def cal_avg_choords(self):
        self.x = sum(self.x_modelwise)/len(self.x_modelwise)
        self.y = sum(self.y_modelwise)/len(self.y_modelwise)
        self.z = sum(self.z_modelwise)/len(self.z_modelwise)
    
class Domain:
    def __init__(self, start_res_id, end_res_id, res_obj_list):
        self.start_res_id= start_res_id
        self.end_res_id= end_res_id
        self.residue_list = res_obj_list
        self.start_res = self.residue_list[0]
        self.end_res = self.residue_list[-1]
        self.type= None
    
    def visualizeV1(self,svg_plane,thickness):
        random_color = self.get_random_color()
        for part in cross_class_functions.divide_list_into_parts(self.residue_list,1):
            start_res = part[0]
            end_res = part[-1]
            svg_plane.add(create_rectangle_between_2_points(start_res.x,start_res.y,end_res.x,end_res.y,random_color,thickness))
        self.add_domain_text(svg_plane)
        
    def visualizeV4(self,svg_plane,thickness):
        random_color = self.get_random_color()
        for part in cross_class_functions.divide_list_into_parts(self.residue_list,1): #if 1: one vis per domain...
            start_point = part[0].x,part[0].y
            end_point = part[-1].x,part[-1].y
            ss_percentages = cross_class_functions.calc_ss_percentages(part)
            objs = self.create_dom_ss_vis(start_point,end_point,ss_percentages,random_color,thickness)
            [svg_plane.add(obj) for obj in objs]
        self.add_domain_text(svg_plane)
    
    def add_domain_text(self,svg_plane):
        ss_percentages = cross_class_functions.calc_ss_percentages(self.residue_list)
        middle_point= self.get_avg_point()
        text = svgwrite.text.Text('helix: '+str(round(ss_percentages[0],2))+', sheet: '+str(round(ss_percentages[1],2))+', coil: '+str(round(ss_percentages[2],2)))
        text.update({'x':str(middle_point[0]),'y':str(middle_point[1]),'font-size': 40})
        svg_plane.add(text)

    def create_dom_ss_vis (self,start_point,end_point, ss_percentages,domain_color, thickness):
        max_value = max(ss_percentages)
        for i in range(0,len(ss_percentages)):
            ss_percentages[i] = ss_percentages[i]/max_value # transfomr percentages to relative percentages to biggest
        #print(ss_percentages)
        middle_point = ((start_point[0]+end_point[0]) /2,(start_point[1] + end_point[1])/2 )
        dx,dy = get_dx_dy(start_point[0],start_point[1],end_point[0],end_point[1])
        length = get_vector_length(dx,dy)
        norm_vec = get_normalized_vector(dx,dy)
        #create sheet:
        sheet_len = ss_percentages[1] * length
        sheet_start_point = move_point_vec(middle_point,norm_vec,-(sheet_len/2))
        sheet_end_point = move_point_vec(middle_point,norm_vec,sheet_len/2)
        sheet = create_arrow_line_between_2_points(sheet_start_point[0],sheet_start_point[1],sheet_end_point[0],sheet_end_point[1],cross_class_functions.lighten_color(domain_color),thickness)
        #create coil:
        coil_len = ss_percentages[2] * length
        coil_start_point = move_point_vec(middle_point,norm_vec,-(coil_len/2))
        coil_end_point = move_point_vec(middle_point,norm_vec,coil_len/2)
        coil = svgwrite.shapes.Polyline(points=[coil_start_point,coil_end_point], stroke=domain_color, stroke_width=10, fill="none")
        #create helix:
        helix_len = ss_percentages[0] * length
        helix_start_point = move_point_vec(middle_point,norm_vec,-(helix_len/2))
        helix_end_point = move_point_vec(middle_point,norm_vec,helix_len/2)
        helix = create_simple_helix_line(helix_start_point[0],helix_start_point[1],helix_end_point[0],helix_end_point[1],cross_class_functions.darken_color(domain_color), thickness, 30, 100)
        return sheet, helix,coil
    
    def get_random_color(self):
        available_colors = ['#FF0000',  # Red
    '#00FF00',  # Green
    '#0000FF',  # Blue
    '#FFFF00',  # Yellow
    '#FF00FF',  # Magenta
    '#00FFFF',  # Cyan
    '#800000',  # Maroon
    '#008000',  # Olive
    '#000080',  # Navy
    '#808000',  # Olive Green
    '#800080',  # Purple
    '#008080',  # Teal
    '#A52A2A',  # Brown
    '#FFA500',  # Orange
    '#808080',  # Gray
    '#C0C0C0',  # Silver
    '#F0F8FF',  # Alice Blue
    '#FFD700'   # Gold
        ]
        return random.choice(available_colors)

    def get_avg_point(self):
        coords=[]
        for res in self.residue_list:
            coords.append((res.x,res.y))
        return cross_class_functions.combine_coords(coords)
    
    def get_string_residues(self):
        string=""
        for residue in self.residue_list:
            string+=residue.amino_acid+", "
        return string
    
    def get_middle_point(self):
    
        mx = (self.start_res.x + self.end_res.x) / 2
        my = (self.start_res.y + self.end_res.y) / 2
        return mx, my

class Protein:
    def __init__(self):
        self.residues = None
        self.secondary_structures = None
        self.best_rotation = None
        self.ordered_vis_elements = None
        self.fam_aligned_parts = None
        self.sf = None
        self.prob = None
        self.chain = None

    def parse_pdb(self,pdb_file):
        self.pdb_file= pdb_file
        parser = PDB.PDBParser(QUIET=True)
        prot_pdb_element =parser.get_structure("protein", pdb_file)
        self.set_obj_residues(prot_pdb_element)
        self.prot_pdb_element =prot_pdb_element
        for residue in self.residues:
            residue.y = -residue.y
            #residue.x = -residue.x
        return prot_pdb_element
    
    def print_ss_objects(self):
        helix=0
        sheets= 0
        coils = 0
        for ss in self.secondary_structures:
            if ss.type == 'helix':
                helix+=1
            elif ss.type == 'sheet':
                sheets+=1
            elif ss.type == 'coil':
                coils+=1
        print(f"helix_elements: {helix}\nsheet_elemts: {sheets}\ncoil_elements: {coils}")
    def set_obj_residues(self,pdb_element):
        atoms_to_save = ['CA','SG']
        residues = {}
        for model in pdb_element:
            res_counter = 1
            all_res_number = 0
            for chain in model:
                for residue in chain:
                    all_res_number+=1
                    res_id = residue.get_resname()+"_"+str(res_counter)
                    for atom in residue:
                        atom_coord = atom.get_coord()
                        atom_name = atom.get_name()
                        x, y, z = atom_coord[0], atom_coord[1], atom_coord[2]
                        if atom_name in atoms_to_save:
                            #print(residue.get_resname())
                            if res_id in residues.keys(): 
                                current_res = residues[res_id]
                            else: 
                                current_res = Residue(residue.get_resname(), res_counter)
                                residues[res_id] = current_res
                            if atom_name in current_res.atoms.keys():
                                curr_atom = current_res.atoms[atom_name]
                                curr_atom.append_model_choords(x,y,z)
                            else:
                                curr_atom = Atom(atom_name,x,y,z)
                                current_res.atoms[atom_name] = curr_atom
                    res_counter+=1
            #print(all_res_number)
        for residue in residues.values():
            residue.set_atoms_average_choords()
        self.residues=list(residues.values())
        return list(residues.values())

    def get_secondary_structure(self, prot_pdb_element, pdb_file):
        residue_ss = cross_class_functions.get_DSSP_SS(prot_pdb_element,pdb_file)
        ss_structure_obj = cross_class_functions.get_ss_blocks(residue_ss,self.residues)
        self.update_residue_object_SS(ss_structure_obj)
        self.secondary_structures = ss_structure_obj

    def update_residue_object_SS(self,ss_objects):
        for ss_obj in ss_objects:
            for res in ss_obj.residue_list:
                res.ss = ss_obj.type
                res.ss_obj = ss_obj

    def save_rot_to_real_coords(self):
        if self.residues[0].x_rot ==None:
            #no rot coords--> no better than original rot found
            return
        for res in self.residues:
            res.x = res.x_rot
            res.y = res.y_rot
            res.z = res.z_rot

    def rotate_by (self,rotation_matrix,direct_safe):
        current_coords = [[res.x,res.y,res.z] for res in self.residues]
        current_coords = np.array(current_coords)
        points_rotated = np.dot(rotation_matrix, current_coords.T).T
        self.save_rot_coords(points_rotated)
        if direct_safe:
            self.save_rot_to_real_coords()
        return points_rotated
    
    def save_rot_coords(self,coords):
        #save per residue
        for i in range(0,len(coords)):
            self.residues[i].x_rot = (coords[i][0]) 
            self.residues[i].y_rot = (coords[i][1]) 
            self.residues[i].z_rot = (coords[i][2] )  

    def move_to_origin(self):
        x_shift,y_shift = self.get_vector_for_origin_shift()
        self.scale_shift_coords(scale=1,x_shift=x_shift,y_shift=y_shift,make_positive=False)

    def get_vector_for_origin_shift(self):
        min_x = min([residue.x for residue in self.residues])
        min_y= min([residue.y for residue in self.residues])
        return (-min_x,-min_y)

    def check_number_overlappong_ss(self,rot_check):
        lines = []
        if rot_check:
            for ss in self.secondary_structures:
                if ss.type != 'coil':
                    line = ((ss.start_res.x_rot,ss.start_res.y_rot),(ss.end_res.x_rot,ss.end_res.y_rot))
                    lines.append(line)
        else:
            for ss in self.secondary_structures:
                if ss.type != 'coil':
                    line = ((ss.start_res.x,ss.start_res.y),(ss.end_res.x,ss.end_res.y))
                    lines.append(line)
        line_objects = [LineString(line) for line in lines]
        intersecting_lines = [(i, j) for i, j in combinations(range(len(lines)), 2)if do_lines_intersect(line_objects[i], line_objects[j])]
        #print("intersecting: "+ str(len(intersecting_lines)))
        return len(intersecting_lines)

    def check_xy_area_of_content_size(self, rot_check, use_coil=False):
        min_x = min_y = min_z =  float('inf')
        max_x = max_y = max_z =  -float('inf')
        only_coil = True
        if rot_check:
            for res in self.residues:
                if res.ss !='coil' or use_coil: #ignore coils for viewpoint finding
                    min_x, min_y= min(min_x,res.x_rot), min(min_y,res.y_rot)
                    max_x, max_y = max(max_x,res.x_rot), max(max_y,res.y_rot)
                    only_coil=False
        else:
            for res in self.residues:
                if res.ss !='coil' or use_coil: #ignore coils for viewpoint finding
                    min_x, min_y= min(min_x,res.x), min(min_y,res.y)
                    max_x, max_y = max(max_x,res.x), max(max_y,res.y)
                    only_coil=False
        if only_coil:
            #print(f"coil: {self.check_xy_area_of_content_size(rot_check, use_coil=True)}")
            return self.check_xy_area_of_content_size(rot_check, use_coil=True)
        return abs(max_x-min_x) * abs(max_y-min_y)

    def check_vis_depth(self, rot_check, use_coil=False):
        min_z = float('inf')     
        max_z = -float('inf')
        only_coil = True
        if rot_check:
            for res in self.residues:
                if res.ss !='coil' or use_coil: 
                    min_z = min(min_z,res.z_rot)
                    max_z = max(max_z,res.z_rot)
                    only_coil=False
        else:
            for res in self.residues:
                if res.ss !='coil' or use_coil: 
                    min_z = min(min_z,res.z)
                    max_z = max(max_z,res.z)
                    only_coil=False
        if only_coil:
            return self.check_vis_depth(rot_check, use_coil=True)
        return abs(max_z-min_z)

    def find_best_view_angle(self,step_width=30):
        theta_range = (0, 360, step_width)
        theta_sets = list(product(range(*theta_range), repeat=3))
        area_rounding_factor= 10

        #factors for choosing rotation:
        best_num_OV = self.check_number_overlappong_ss(rot_check=False)
        best_xy_content_area = round(self.check_xy_area_of_content_size(rot_check=False) / area_rounding_factor) * area_rounding_factor 
        best_z_depth = self.check_vis_depth(rot_check=False)

        print("Start optimal viewpoint calculation: ")
        counter = 0
        rot_protein = copy.deepcopy(self)
        for theta_set in tqdm(theta_sets, desc="Testing rotations", unit="rot"):
            theta_x_rad, theta_y_rad, theta_z_rad = np.radians(theta_set)
            
            rotation_matrix = cross_class_functions.radians_to_rotation_matrix(theta_x_rad, theta_y_rad, theta_z_rad)
            points_rotated = rot_protein.rotate_by(rotation_matrix, direct_safe=False)
        
            current_xy_content_area =round(rot_protein.check_xy_area_of_content_size(rot_check=True) / area_rounding_factor) * area_rounding_factor
            current_num_OV = rot_protein.check_number_overlappong_ss(rot_check=True)
            current_z_depth = rot_protein.check_vis_depth(rot_check=True)
            #print(current_xy_content_area,current_num_OV)
            #if (current_num_OV < best_num_OV) or (current_num_OV == best_num_OV and current_xy_content_area > best_xy_content_area):
            if (current_xy_content_area > best_xy_content_area) or (current_xy_content_area == best_xy_content_area and current_num_OV < best_num_OV) or (current_xy_content_area == best_xy_content_area and current_num_OV == best_num_OV and current_z_depth< best_z_depth):
                #better rotatoin found: save as rot coords in og protein
                self.save_rot_coords(points_rotated)
                best_xy_content_area = current_xy_content_area
                best_num_OV = current_num_OV
                best_z_depth = current_z_depth
                #print(rotation_matrix)
                self.best_rotation = rotation_matrix
            counter+=1
            #print(str(counter) + "/"+str(len(theta_sets)))
        self.save_rot_to_real_coords()
        return self.residues

    def get_cystein_bonds(self,max_length=3):
        cysteines = [res for res in self.residues if res.amino_acid == 'CYS']
        pairs = list(tqdm(combinations(cysteines, 2), desc="Generating cystein pairs", unit="pair"))
        cysteine_bonds = [poss_bond for poss_bond in tqdm(pairs, desc="Checking cystein bonds", unit="bond") if poss_bond[0].atoms['SG'].get_distance_to_atom(poss_bond[1].atoms['SG']) <= max_length]
        return cysteine_bonds

    def scale_shift_coords(self,scale, x_shift,y_shift, make_positive):
        if make_positive:
            min_x = min_y =  float('inf')
            for res in self.residues:
                min_x = min(min_x,res.x)
                min_y = min(min_y,res.y)
            for res in self.residues:
                res.x,res.y = (res.x-min_x) *scale +x_shift,(res.y-min_y) *scale +y_shift
        else:
            for res in self.residues:
                res.x,res.y = (res.x) *scale +x_shift,(res.y) *scale +y_shift
    
    def get_domains(self,domain_annotation_file):
        domain_list = []
        with open(domain_annotation_file, 'r') as input_file:
            header = input_file.readline().strip()
            lines = input_file.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            domain_start = int(parts[2])
            domain_end = int(parts[3])
            domain_res_list = []
            for res in self.residues:
                if domain_start <=res.chain_pos <=domain_end:
                    domain_res_list.append(res)
            #domain_list.append(Domain(domain_start,domain_end, res_list))
            domain_secondary_structures = []
            for res in domain_res_list:
                if res.ss_obj not in domain_secondary_structures:
                    domain_secondary_structures.append(res.ss_obj)
            #chekc first and last ss for domain overlap..
            delete_until = 0
            for i in range(0,len(domain_secondary_structures[0].residue_list)):
                res = domain_secondary_structures[0].residue_list[i]
                if res.chain_pos < domain_res_list[0].chain_pos: # part of ss not included in domain
                    delete_until = i+1
            corrected_ss = Secondary_Structure(domain_secondary_structures[0].type,domain_secondary_structures[0].residue_list[delete_until:])
            domain_secondary_structures[0] = corrected_ss

            delete_from = len(domain_secondary_structures[-1].residue_list)
            for i in range(0,len(domain_secondary_structures[-1].residue_list)):
                res = domain_secondary_structures[-1].residue_list[i]
                if res.chain_pos > domain_res_list[-1].chain_pos: # part of ss not included in domain
                    delete_from = i
                    break
            corrected_ss = Secondary_Structure(domain_secondary_structures[-1].type,domain_secondary_structures[-1].residue_list[:delete_from])
            domain_secondary_structures[-1] = corrected_ss               

            domain_protein = Protein()
            domain_protein.residues= domain_res_list
            domain_protein.secondary_structures = domain_secondary_structures
            domain_list.append(domain_protein)
        return domain_list
    
    def get_all_coords (self):
        coords = []
        for res in self.residues:
            coords.append([res.x,res.y])
        return coords
    
    def add_hull(self,svg_plane,color, opacity,smoothness=0.1):
        points = np.array(self.get_all_coords())
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        coords =[]
        [coords.append((point[0],point[1])) for point in hull_points]
        coords.append(coords[0])
        
        #smooth line
        tck, u = splprep(np.array(coords).T, s=smoothness, per=True)
        new_points = splev(np.linspace(0, 1, 1000), tck, der=0)
        smoothed_coords = list(zip(new_points[0], new_points[1]))
        #add to svg
        svg_plane.add(svgwrite.shapes.Polyline(points=smoothed_coords,stroke='black',stroke_width=10,fill=color,opacity=opacity))
          
    def add_sf_info(self,svg_plane):
        points = np.array(self.get_all_coords())
        max_y = np.max(points[:, 1])
        center_x = np.median(points[:, 0])
        
        svg_plane.add(svgwrite.text.Text(f"{self.sf} ({self.prob})", insert=(center_x-130, max_y + 100),fill='black', font_size='80'))  
    def add_chain_info(self,svg_plane):
        points = np.array(self.get_all_coords())
        min_x = np.min(points[:, 0])
        center_y = np.median(points[:, 1])
        
        svg_plane.add(svgwrite.text.Text(f"{self.chain}", insert=(min_x -100, center_y),fill='black', font_size='80'))  
    
    def get_hull_color(self):
        #calc hull color rgb based on ss percentages 
        rgb_range = 255
        helix_count,sheet_count,coil_count = 0,0,0
        for residue in self.residues:
            if residue.ss == "helix":
                helix_count += 1
            elif residue.ss == "sheet":
                sheet_count += 1
            elif residue.ss == "coil":
                coil_count += 1
        total_residues = len(self.residues)
        helix_percentage,sheet_percentage,coil_percentage= (helix_count / total_residues),(sheet_count / total_residues),(coil_count / total_residues)
        
        #calc rgb color based on percentages (helix=red, coil=blue, sheet=green)
        helix_color,sheet_color, coil_color = int(helix_percentage*rgb_range), int(sheet_percentage*rgb_range), int(coil_percentage*rgb_range)
        return svgwrite.rgb(*(helix_color,coil_color,sheet_color))

    def get_average_point(self):
        x_sum = 0
        y_sum = 0
        for res in self.residues:
            x_sum+= res.x
            y_sum+= res.y
        return x_sum/len(self.residues),y_sum/len(self.residues)
    
    def recalc_positions_based_based_on_new_avg(self,new_avg):
        original_avg_point = self.get_average_point()
        dx,dy = get_dx_dy(original_avg_point[0],original_avg_point[1],new_avg[0],new_avg[1])
        norm_vec = get_normalized_vector(dx,dy)
        distance = get_vector_length(dx,dy)
        for res in self.residues:
            res.x,res.y = move_point_vec((res.x,res.y),norm_vec,distance)

    def overlaps_with_dom(self,second_domain):

        x1 = [residue.x for residue in self.residues]
        y1 = [residue.y for residue in self.residues]
        x2 = [residue.x for residue in second_domain.residues]
        y2 = [residue.y for residue in second_domain.residues]
        
        overlap = (
            min(x1) < max(x2) and max(x1) > min(x2) and
            min(y1) < max(y2) and max(y1) > min(y2)
        )
        return overlap
    
    def init_empty(self):
        self.secondary_structures = []
        self.residues = []

    def split_aligned_part(self,aligned_segment):

        front_part = Protein()
        front_part.init_empty()
        aligned_part = Protein()
        aligned_part.init_empty()
        end_part = Protein()
        end_part.init_empty()

        for ss in self.secondary_structures:
            in_protein = None
            
            if ss.start_res.chain_pos < aligned_segment[1]-1 and  aligned_segment[0]-1 < ss.end_res.chain_pos:
                #ss is in a aligned part
                in_protein = aligned_part
            elif ss.end_res.chain_pos <= aligned_segment[0]-1:
                #ss in front part
                in_protein = front_part
            else:
                # ss in end part
                in_protein = end_part
            in_protein.secondary_structures.append(ss)
            [in_protein.residues.append(residue) for residue in ss.residue_list]
        
        self.fam_aligned_parts = front_part,aligned_part,end_part
        return front_part,aligned_part,end_part
    
    def connect_to_protein_dashline(self,second_protein):
        if len(self.residues) ==0 or len(second_protein.residues) ==0:
            return None
        dasharray = "15,10"
        start_res = self.residues[-1] if self.residues[-1].included_in == None else self.residues[-1].included_in
        end_res = second_protein.residues[0] if second_protein.residues[0].included_in == None else second_protein.residues[0].included_in
        dash_line = svgwrite.shapes.Line(start=(start_res.x, start_res.y), end=(end_res.x, end_res.y), stroke='black',fill='none', stroke_width=5, stroke_dasharray=dasharray)
        return dash_line
    
    def get_protein_ordered_vis_objects(self, avg_coil, mark_endings):
        if len(self.residues) == 0:
            return
        #returns the z-ordered objects for visualisation in a list (helix,sheet,coil, connecting_elements, cystein_bonds)
        vis_object_list = []

        #add secondary structures and connectin elements
        last_ss = None
        for ss in self.secondary_structures:
            if ss.type =='coil':
                ss.coil_path = ss.get_average_path(avg_coil)
            vis_object_list.append(ss)
            if last_ss !=None:
                connect1 = last_ss.end_res.included_in if last_ss.end_res.included_in != None else last_ss.end_res
                connect2 = ss.start_res.included_in if ss.start_res.included_in != None else ss.start_res
                connect_elemet = Connecting_element(connect1,connect2)
                vis_object_list.append(connect_elemet)
            last_ss = ss
        #add cystein bonds
        bonds = self.get_cystein_bonds()
        for bond in bonds:
            cys_bond = Cystein_bond(bond[0],bond[1])
            vis_object_list.append(cys_bond)

        #add lddt_res
        created_lddt_list = []
        for res in self.residues:
            if res.included_in != None:
                #is summarise
                sum_point =res.included_in
                if (sum_point.x,sum_point.y) not in created_lddt_list: #no double lddt point drawing
                    lddt_obj=Lddt_coloring_res(sum_point.x,sum_point.y,sum_point.z, sum_point.lddt)
                    created_lddt_list.append((sum_point.x,sum_point.y))
                    vis_object_list.append(lddt_obj)
            elif res.lddt != None:
                #get closest point on line for helix/sheet
                line_x,line_y = closest_point_on_line((res.x,res.y),res.ss_obj.get_line()) if res.ss!='coil' else (res.x,res.y)
                lddt_obj=Lddt_coloring_res(line_x,line_y,res.ss_obj.z,res.lddt)
                vis_object_list.append(lddt_obj)
                
            
        if mark_endings:
            #add annotation (start,end)
            start_res = self.residues[0] if self.residues[0].included_in== None else self.residues[0].included_in
            end_res = self.residues[-1] if self.residues[-1].included_in== None else self.residues[-1].included_in
            start_annotation = Annotation(start_res,'start','blue')
            end_annotation = Annotation(end_res,'end','blue')
            vis_object_list.append(start_annotation)
            vis_object_list.append(end_annotation)

        #sort list
        vis_object_list.sort(key=lambda obj: obj.z, reverse=True)
        self.ordered_vis_elements= vis_object_list
        return vis_object_list

    def draw_simplified_path(self,svg_plane, averaging, opacity):      
        avg_path = []
        avg_path.append((self.residues[0].x,self.residues[0].y)) # start fix
        
        for i in range(0, len(self.residues), averaging):
            residue_segment = self.residues[i:i+averaging]
            sum_point=cross_class_functions.calc_summarising_point(residue_segment)
            avg_path.append((sum_point.x,sum_point.y))
        avg_path.append((self.residues[-1].x,self.residues[-1].y)) # end fix
        
        svg_plane.add(svgwrite.shapes.Polyline(points=avg_path, stroke='black', stroke_width=10, fill="none",opacity=opacity))
        return avg_path

    def add_lddt_to_aligned_residues(self,aligned_region,lddtfull):
        lddt_map = {aligned_region[0] + i: lddt for i, lddt in enumerate(lddtfull)}
        for res in self.residues:
            if res.chain_pos in lddt_map:
                res.lddt = float(lddt_map[res.chain_pos])

class Cystein_bond:
    def __init__(self,start_res,end_res):
        self.type = 'cystein_bond'
        self.start_res = start_res
        self.end_res = end_res
        # use highes point of bond as order value --> always on top of ss
        self.z = min(self.start_res.z,self.end_res.z)
        #self.z = (self.start_res.z+self.end_res.z) /2

class Connecting_element:
    def __init__(self,start_res,end_res):
        self.type = 'connecting_element'
        self.start_res = start_res
        self.end_res = end_res
        self.z = max(self.start_res.z,self.end_res.z)
        #TODO maybe change to SS "z"...always at the same level as connected SS

class Lddt_coloring_res:
    def __init__(self,x,y,z,lddt):
        self.type = 'lddt_coloring_res'
        self.x,self.y,self.z = x,y,z
        self.lddt = lddt

class Annotation:
    def __init__(self,residue,text,color):
        self.type='annotation'
        self.annotated_residue = residue
        self.text = text
        self.z = residue.z
        self.color = color