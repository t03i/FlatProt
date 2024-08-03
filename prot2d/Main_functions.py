import svgwrite
from itertools import combinations
import networkx as nx
import itertools
from Bio.PDB import PDBParser, PDBIO
import os
import csv

from .Classes import *
from .Graphical import*

def split_pdb_chains(input_pdb_file, output_dir):
    """
    Splits a PDB file into multiple files based on chains using Biopython, saves them to a specified directory, 
    and returns the paths. If only one chain is found, returns the path of the original PDB file instead.
    Args:
    input_pdb_file (str): The path to the input PDB file.
    output_dir (str): The directory where the output files will be saved.
    Returns:
    list of str: The paths to the generated PDB files or the original file if only one chain is present.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdb_parser = PDB.PDBParser()
    structure = pdb_parser.get_structure('structure_name', input_pdb_file)
    
    io = PDB.PDBIO()
    file_paths = {}

    for model in structure:
        for chain in model:
            chain_id = chain.id
            output_file_name = f"{os.path.splitext(os.path.basename(input_pdb_file))[0]}_chain_{chain_id}.pdb"
            output_file_path = os.path.join(output_dir, output_file_name)
            io.set_structure(chain)
            io.save(output_file_path)
            file_paths[chain_id]= output_file_path
    return file_paths

def build_domainpdb_from_fullpdb(pdb_file, domain_start, domain_end,domain_index, output_dir):
    base_name = os.path.basename(pdb_file)
    file_id = os.path.splitext(base_name)[0]
    output_file = os.path.join(output_dir, f"{file_id}_domain_{str(domain_index)}.pdb")

    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    for model in structure:
        for chain in model:
            for residue in list(chain):
                if residue.id[1] < domain_start or residue.id[1] > domain_end:
                    chain.detach_child(residue.id)

    
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)
    return output_file

def get_domain_pdbs(pdb_file,chain_id, annotation_file, output_dir):
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
    print(f"pdb id: {pdb_id}")
    domain_files = []
    with open(annotation_file, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            if row['chain_id'] == pdb_id:
                domain_start = int(row['domain_start'])
                domain_end = int(row['domain_end'])
                domain_index = int(row['domain'])
                dom_file = build_domainpdb_from_fullpdb(pdb_file, domain_start, domain_end, domain_index, output_dir)
                domain_files.append(dom_file)
    return domain_files

def add_header_to_pdb(pdb_file):
    header_line = "added    for DSSP"
    with open(pdb_file, 'r') as input_file:
        pdb_content = input_file.read()

    # Check if the header already exists
    if not pdb_content.startswith("HEADER"):
        # Header does not exist, add it
        updated_pdb_content = f"HEADER    {header_line}\n{pdb_content}"

        with open(pdb_file, 'w') as output_file:
            # Write the updated content back to the file
            output_file.write(updated_pdb_content)
        
        return output_file, True  # Indicate that the header was added
    else:
        # Header already exists
        return pdb_file, False  # Indicate that the header was not added

def shift_domains_in_x_line(domain_proteins, dom_gap):
    if len(domain_proteins) >1:
        for dom_prot in domain_proteins:   dom_prot.move_to_origin()
        last_dom_prot = domain_proteins[0]
        for dom_prot in domain_proteins[1:]:
            x_overlap=calc_x_overlap_distance_between_prots(last_dom_prot,dom_prot)
            dom_prot.scale_shift_coords(scale=1,x_shift=x_overlap+dom_gap,y_shift=0, make_positive=False)
            last_dom_prot=dom_prot
    else:
        domain_proteins[0].move_to_origin()

def shift_chains_in_y_line(chain_prots, chain_gap):
    if len(chain_prots) >1:
        for chain_prot in chain_prots:   chain_prot.move_to_origin()
        last_chain_prot = chain_prots[0]
        for chain_prot in chain_prots[1:]:
            y_overlap=calc_y_overlap_distance_between_prots(last_chain_prot,chain_prot)
            chain_prot.scale_shift_coords(scale=1,x_shift=0,y_shift=y_overlap+chain_gap, make_positive=False)
            last_chain_prot=chain_prot
    else:
        chain_prots[0].move_to_origin()

def calc_x_overlap_distance_between_prots(prot1,prot2):
    if len(prot1.residues) == 0 or len(prot2.residues)==0:
        return 0
    prot1_max_x = max(residue.x for residue in prot1.residues)
    prot2_min_x = min(residue.x for residue in prot2.residues)
    overlap = prot1_max_x - prot2_min_x
    return overlap if overlap>0 else 0
def calc_y_overlap_distance_between_prots(prot1,prot2):
    if len(prot1.residues) == 0 or len(prot2.residues)==0:
        return 0
    prot1_max_y = max(residue.y for residue in prot1.residues)
    prot2_min_y = min(residue.y for residue in prot2.residues)
    overlap = prot1_max_y - prot2_min_y
    return overlap if overlap>0 else 0
def add_dashed_line_between_proteins(dwg,prot1,prot2,prot3):
    dash_line = prot1.connect_to_protein_dashline(prot2)
    if dash_line is not None:
        dwg.add(dash_line)
    dash_line2 = prot2.connect_to_protein_dashline(prot3)
    if dash_line2 is not None:
        dwg.add(dash_line2)


def get_min_coords(prot_structure):
    min_x = min_y = min_z =  float('inf')
    for model in prot_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "CA":
                        atom_coord = atom.get_coord()
                        x, y, z = atom_coord[0], atom_coord[1], atom_coord[2]
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        min_z = min(min_z, z)
    return min_x, min_y, min_z
    

def simplify_DSSP_output(secondary_structure_string):
    # convert 8 types to 3
    three_type_ss_string = secondary_structure_string.replace('G','H') #Helix-3 to Helix
    three_type_ss_string = three_type_ss_string.replace('I','H') #Helix-5 to Helix
    three_type_ss_string = three_type_ss_string.replace('B','E') #Beta_bridge to Strand
    return three_type_ss_string

def calc_residue_coordinates_dict (prot_structure,scale_factor, slide_factor):
    # not used!!!
    min_x, min_y, min_z = get_min_coords(prot_structure)
    
    #add coordinates of wanted model (using scale factor + make everything positive)
    res_counter = 0 #needed beacuse dsssp SS is identified starting with 1
    residue_coords_array =[]
    residue_ASs_array = []
    for model in prot_structure:
        residue_coords ={}
        residue_ASs = {}
        for chain in model:
            for residue in chain:
                res_counter+=1
                for atom in residue:
                    if atom.get_name() == "CA":  
                        atom_coord = atom.get_coord()
                        x, y = atom_coord[0], atom_coord[1]
                        x = (x - min_x) * scale_factor + slide_factor
                        y = (y - min_y) * scale_factor + slide_factor
                        residue_coords[res_counter] = (str(x), str(y))
                        residue_ASs[res_counter] = residue.get_resname()
        residue_coords_array.append(residue_coords)
        residue_ASs_array.append(residue_ASs)
    avg_residue_coords = calc_model_choord_average(residue_coords_array, residue_ASs_array[0])
    # residue_Aas from model 0 will be used as referecne later on --> needed for DSSP usages
    return avg_residue_coords, residue_ASs_array[0]

def calc_model_choord_average(residue_coords_array, residue_ASs):
    # Initialize a dictionary to store the average coordinates for each residue
    average_coords = {}
    # Iterate through each of the five dictionaries
    for residue_counter in residue_ASs.keys():
        total_x = 0.0
        total_y = 0.0
        count = 0
        # Calculate the sum of coordinates and count for the current residue across all dictionaries
        for coord_dict in residue_coords_array:
            if len(coord_dict) != len(residue_coords_array[0]):
                print("model not valid!")
                continue
            if residue_counter in coord_dict:
                x, y = map(float, coord_dict[residue_counter])
                total_x += x
                total_y += y
                count += 1
        # Calculate the average coordinates for the current residue
        if count > 0:
            avg_x = total_x / count
            avg_y = total_y / count
            average_coords[residue_counter] = (str(avg_x), str(avg_y))
        else:
            print("calc_model_coord_average ERROR")
    return average_coords

def create_SS_polyLines(svg_plane,blocks_coords, line_color, line_width):
    for ss_segment_coords in blocks_coords:
        segment_polyline = svgwrite.shapes.Polyline(points=ss_segment_coords, stroke=line_color, stroke_width=line_width, fill="none")
        svg_plane.add(segment_polyline)

def visualize_ordered_elements(svg_plane,ordered_vis_elements, simple_helix, general_opacity, cysteins, show_lddt_col,show_lddt_text, no_coil=False,shadow_col=False):
    if shadow_col:
        helix_col = 'grey'
        sheet_col = 'grey'
        coil_col = 'grey'
        cystein_col = 'grey'
    else:
        helix_col = '#F05039'
        sheet_col = '#1F449C'
        coil_col = 'black'
        cystein_col = 'orange'
    circle_radius= 3.5
    cys_annotation = False
    for element in ordered_vis_elements:
        if element.type =='sheet':
            x1,y1 = element.start_res.x,element.start_res.y
            x2,y2 = element.end_res.x, element.end_res.y
            svg_plane.add(create_arrow_line_between_2_points(float(x1),float(y1),float(x2),float(y2),sheet_col, 80, general_opacity))
            #add circle for ss change
            svg_plane.add(svgwrite.shapes.Circle(center=(str(x1),str(y1)), r=circle_radius,fill='black', stroke='black', opacity=general_opacity))
            svg_plane.add(svgwrite.shapes.Circle(center=(str(x2),str(y2)), r=circle_radius,fill='black', stroke='black', opacity=general_opacity))
        elif element.type =='helix':
            x1,y1 = element.start_res.x,element.start_res.y
            x2,y2 = element.end_res.x, element.end_res.y
            #print(f"({x1,y1})({x2,y2})")
            #svg_plane.add(svgwrite.shapes.Circle(center = (x1,y1),r=20, fill="orange", opacity=1))
            #svg_plane.add(svgwrite.shapes.Circle(center = (x2,y2),r=20, fill="orange", opacity=1))
            #svg_plane.add(svgwrite.shapes.Polyline(points=[(x1,y1),(x2,y2)], stroke="orange", stroke_width=5, fill="none", opacity=1))
            if(simple_helix):
                svg_plane.add(create_simple_helix_line(float(x1),float(y1),float(x2),float(y2), helix_col, 100, 50, 100, general_opacity))
            else:
                svg_plane.add(create_helix_between(float(x1),float(y1),float(x2),float(y2), helix_col, 80))
            #print(f"\n{x1}\n")
            svg_plane.add(svgwrite.shapes.Circle(center=(str(x1),str(y1)), r=circle_radius,fill='black', stroke='black'))
            svg_plane.add(svgwrite.shapes.Circle(center=(str(x2),str(y2)), r=circle_radius,fill='black', stroke='black'))
        elif element.type == 'coil' and not no_coil:
            points = [(str(point[0]), str(point[1])) for point in element.coil_path]
            svg_plane.add(svgwrite.shapes.Polyline(points=points, stroke=coil_col, stroke_width=5, fill="none", opacity=general_opacity))

        elif element.type == 'connecting_element' and not no_coil:
            svg_plane.add(svgwrite.shapes.Polyline(points=[(str(element.start_res.x),str(element.start_res.y)),(str(element.end_res.x),str(element.end_res.y))], stroke='black', stroke_width=5, fill="none", opacity=general_opacity))

        elif element.type == 'cystein_bond' and cysteins and not no_coil:
            
            x1,y1 = element.start_res.get_closest_point()
            x2,y2 = element.end_res.get_closest_point()
            svg_plane.add(svgwrite.shapes.Line(start=(x1,y1),end=(x2,y2), stroke=cystein_col, stroke_width=3, fill="none",opacity=general_opacity, stroke_dasharray='2,2'))
            svg_plane.add(svgwrite.shapes.Circle(center=(str(x1),str(y1)), r=circle_radius,fill=cystein_col, stroke=cystein_col,opacity=general_opacity))
            svg_plane.add(svgwrite.shapes.Circle(center=(str(x2),str(y2)), r=circle_radius,fill=cystein_col, stroke=cystein_col,opacity=general_opacity))
            if cys_annotation:
                cyst_text = svgwrite.text.Text(element.start_res.amino_acid +" : " +str(element.start_res.chain_pos) +" -- "+element.end_res.amino_acid +" : " +str(element.end_res.chain_pos))
                cyst_text.update({'x':str(x1),'y':str(y1)})
                svg_plane.add(cyst_text)
        elif element.type == 'annotation':
            dot = svgwrite.shapes.Circle(center = (str(element.annotated_residue.x),str(element.annotated_residue.y)),r=5, fill=element.color )
            text = svgwrite.text.Text(element.text, fill =element.color)
            text.update({'x':str(element.annotated_residue.x+5),'y':str(element.annotated_residue.y+10)})
            svg_plane.add(dot)
            svg_plane.add(text)

        elif element.type == 'lddt_coloring_res':
            if element.lddt ==None:
                continue
            text = svgwrite.text.Text(round(element.lddt,3), fill ="black")
            text.update({'x':str(element.x+10),'y':str(element.y+10)})
            lddt_color = get_lddt_color(element.lddt)
            
            svg_plane.add(text) if show_lddt_text and lddt_color!=None else None
            svg_plane.add(svgwrite.shapes.Circle(center = (str(element.x),str(element.y)),r=15, fill=lddt_color, opacity=0.3)) if show_lddt_col and lddt_color!=None  else None

def create_testing_vis(ssv, svg_plane, ss_objects):
    for ss in ss_objects:
        if ss.type != 'coil':
            x1,y1 = ss.start_res.x,ss.start_res.y
            x2,y2 = ss.end_res.x, ss.end_res.y
            svg_plane.add(svgwrite.shapes.Polyline(points=[(x1,y1),(x2,y2)], stroke='black', stroke_width=5, fill="none"))

def get_lddt_color(value):
    if value > 0.5:
        return None
    value_scaled = 1- value * 2
    r = b = int(128 + (127 * value_scaled))  # 128 is the start for dark magenta, add up to 127 based on value
    g = 0
    return f'#{r:02X}{g:02X}{b:02X}'

def do_fruchtermann_reingold_layout(ss_objects,k,iter):
    G = nx.Graph()
    nodes = {}
    for ss in ss_objects:
        if ss.type !='coil':
            G.add_node(ss.start_res, pos= (ss.start_res.x, ss.start_res.y))
            G.add_node(ss.end_res,pos= (ss.end_res.x, ss.end_res.y))
            nodes[ss.start_res] = (ss.start_res.x, ss.start_res.y)
            nodes[ss.end_res]=(ss.end_res.x, ss.end_res.y)
    #creat fixed notes:
    fixed_nodes=[]
    for ss in ss_objects:
        if ss.type !='coil':
            fixed_nodes.append(ss.start_res)
            break
    print("fixed: ")
    print(fixed_nodes)
    #calc and save new points
    new_pos = nx.fruchterman_reingold_layout(G,pos=nodes,k=k,iterations=iter, fixed=fixed_nodes)
    for ss in ss_objects:
        if ss.type !='coil':
            ss.start_res.x,ss.start_res.y = new_pos[ss.start_res]
            ss.end_res.x, ss.end_res.y = new_pos[ss.end_res]

def do_domain_fruchterman_layout(domains,k,iter):
    G = nx.Graph()
    nodes = {}
    for dom in domains:
        G.add_node(dom, pos= dom.get_average_point())
        nodes[dom] = dom.get_average_point()
    fixed_nodes=[domains[0]]

    new_averages = nx.fruchterman_reingold_layout(G,pos=nodes,k=k,iterations=iter, fixed=fixed_nodes)
    for dom in new_averages:
        dom.recalc_positions_based_based_on_new_avg(new_averages[dom])


def create_sheet_representations(ssv, svg_plane,sheet_blocks_coords):
    for block in sheet_blocks_coords:
        x1,y1 = block[0]
        x2,y2 = block[-1]
        svg_plane.add(ssv.create_arrow_line_between_2_points(float(x1),float(y1),float(x2),float(y2),'green', 80))
def create_helix_representation(ssv, svg_plane, helix_blocks_coords, simple):
     for block in helix_blocks_coords:
        x1,y1 = block[0]
        x2,y2 = block[-1]
        if(simple):
            svg_plane.add(ssv.create_simple_helix_line(float(x1),float(y1),float(x2),float(y2), 'red', 100, 50, 100))
        else:
            svg_plane.add(ssv.create_helix_between(float(x1),float(y1),float(x2),float(y2), 'red', 80))
def get_residue_list(ss_structure_obj):  
    residue_list = []
    last_residue = None
    for ss in ss_structure_obj:
        for res in ss.residue_list:
            if not res.equals_residue(last_residue): #doblets because in case of ss-segment change the residue is in both segments for vis
                residue_list.append(res)
                last_residue = res  
    return residue_list
def add_residue_annotations(svg_plane, ss_structures, only_path):
    residue_list = get_residue_list(ss_structures)
    for residue in residue_list:
        if residue.ss != 'coil' and not only_path:
            structure_middle_point = residue.ss_obj.get_middle_point()
            svg_plane.add(svgwrite.shapes.Circle(center=(structure_middle_point[0],structure_middle_point[1]), r=5, fill="black"))
            AS_text = svgwrite.text.Text(residue.ss+": "+residue.ss_obj.get_string_residues())
            AS_text.update({'x':structure_middle_point[0],'y':structure_middle_point[1]})
            svg_plane.add(AS_text)

        else:
            if residue.included_in != None:
                sum_point = residue.included_in
                svg_plane.add(svgwrite.shapes.Circle(center=(sum_point.x,sum_point.y), r=5, fill="black"))
                AS_text = svgwrite.text.Text(sum_point.get_string_residues())
                AS_text.update({'x':sum_point.x,'y':str(float(sum_point.y)-5)})
                svg_plane.add(AS_text)
            else:
                svg_plane.add(svgwrite.shapes.Circle(center=(residue.x,residue.y), r=5, fill="black"))
                AS_text = svgwrite.text.Text(residue.amino_acid)
                AS_text.update({'x':residue.x,'y':str(float(residue.y)-5)})
                svg_plane.add(AS_text)

def check_length_ss_segments(ss_obj):
    overall_length= 0
    for ss in ss_obj:
        if ss.type != "coil":
            ss_length = get_coord_distance(ss.start_res.x,ss.start_res.y,ss.end_res.x,ss.end_res.y)
            overall_length+=ss_length
    return overall_length

   
def add_start_end_annotation(svg_plane,ss_objects, start_col, end_col):
    residue_list = get_residue_list(ss_objects)
    start_res = residue_list[0]
    end_res = residue_list[-1]
    start_dot = svgwrite.shapes.Circle(center = (start_res.x,start_res.y),r=5, fill=start_col )
    start_text = svgwrite.text.Text("start", fill =start_col)
    start_text.update({'x':start_res.x,'y':start_res.y+10})
    
    end_dot = svgwrite.shapes.Circle(center= (end_res.x,end_res.y),r=5, fill=end_col )
    end_text = svgwrite.text.Text("end", fill=end_col)
    end_text.update({'x':end_res.x,'y':end_res.y+10})
    
    svg_plane.add(start_text)
    svg_plane.add(end_text)
    svg_plane.add(start_dot)
    svg_plane.add(end_dot)

def get_cystein_bonds_old(residues, max_length=3):
    #not used
    cysteines = []
    [cysteines.append(res) if res.amino_acid=='CYS' else None for res in residues]
    
    pairs = list(combinations(cysteines, 2))
    cysteine_bonds = []
    [cysteine_bonds.append(poss_bond) for poss_bond in pairs if poss_bond[0].atoms['SG'].get_distance_to_atom(poss_bond[1].atoms['SG']) <= max_length]
    return cysteine_bonds

def vis_cystein_bonds(svg_plane, cystein_bonds, opacity, annotation):
    circle_radius = 6
    for bond in cystein_bonds:
        #find closest point on line to residue in bond and draw yellow line
        x1,y1 = bond[0].get_closest_point()
        x2,y2 = bond[1].get_closest_point()
        svg_plane.add(svgwrite.shapes.Polyline(points=[(x1,y1),(x2,y2)], stroke='orange', stroke_width=3, fill="none",opacity=opacity))
        svg_plane.add(svgwrite.shapes.Circle(center=(x1,y1), r=circle_radius,fill='none', stroke='orange',opacity=opacity))
        svg_plane.add(svgwrite.shapes.Circle(center=(x2,y2), r=circle_radius,fill='none', stroke='orange',opacity=opacity))
        if annotation:
            cyst_text = svgwrite.text.Text(bond[0].amino_acid +" : " +str(bond[0].chain_pos) +" -- "+bond[1].amino_acid +" : " +str(bond[1].chain_pos))
            cyst_text.update({'x':x1,'y':y1})
            svg_plane.add(cyst_text)
        


def calculate_viewbox(objects, padding):
    min_x = min(res.x for res in objects)
    min_y = min(res.y for res in objects)
    max_x = max(res.x for res in objects)
    max_y = max(res.y for res in objects)
    #print(min_x,max_x,min_y,max_y)
    #add some padding to ensure all objects are visible
    return f"{min_x - padding} {min_y - padding} {max_x - min_x + 2 * padding} {max_y - min_y + 2 * padding}"

def connect_domains(domains,svg_plane,color,thickness,opacity):
    last_dom = None
    for dom in domains:
        if last_dom != None:
            start_point = last_dom.residues[-1].x,last_dom.residues[-1].y
            end_point = dom.residues[0].x,dom.residues[0].y
            svg_plane.add(svgwrite.shapes.Polyline(points=[start_point,end_point], stroke=color, stroke_width=thickness, fill="none",opacity=opacity))
        last_dom=dom

def repeat_layout(domains, k=1.5, iter_steps=20):
    overlap = True
    while overlap:
        overlap = False
        for dom1, dom2 in itertools.combinations(domains, 2):
            if dom1.overlaps_with_dom(dom2):
                overlap = True
                break  
        if overlap:
            do_domain_fruchterman_layout(domains, k=k, iter=iter_steps)
def repeat_layout_pairwise(domains, k=1.5, iter_steps=20):
    if len(domains)==1:
        return
    while True:
        overlapping_domains = None
        for dom1, dom2 in itertools.combinations(domains, 2):
            if dom1.overlaps_with_dom(dom2):
                overlapping_domains = (dom1, dom2)
                break
        if not overlapping_domains:
            break
        do_domain_fruchterman_layout(overlapping_domains, k=k, iter=iter_steps)


