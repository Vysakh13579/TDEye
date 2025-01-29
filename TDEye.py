import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from cleaning import clean_data, get_lightcurve_data, inverse_variance_weighting

import glob
import os


plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['lines.markersize'] = 3 

global colors
colors = {'ZTF_r':'darkred', 'ZTF_g':'darkgreen', 'ZTF_i':'indigo',
            'UVW2.uvot':'darkviolet', 'UVM2.uvot':'magenta', 'UVW1.uvot':'fuchsia',
            'B.uvot':'dodgerblue', 'U.uvot':'mediumblue', 'V.uvot':'coral' }

for surv in ['ps','sdss']:
  colors['g.'+surv] = 'g'
  colors['r.'+surv] = 'r'
  colors['i.'+surv] = 'brown'

colors['UVW2.uvot'] = 'violet'
colors['UVM2.uvot'] = 'magenta'
colors['UVW1.uvot'] = 'fuchsia'

colors['U.uvot'] = 'darkblue'
colors['u.sdss'] = 'darkblue'

colors['F125LP'] = 'darkviolet'
colors['F150LP'] = 'darkviolet'
colors['F225W'] = 'magenta'

colors['FUV'] = 'darkviolet'
colors['NUV'] = 'magenta'

colors['B.uvot'] = 'lightblue'
colors['V.uvot'] = 'orange'
colors['c.atlas'] = 'cyan'
colors['o.atlas'] = 'orange'

#########################################################
#####    USEFUL funtions
#########################################################

def get_filename_and_id(ID, df):
    """
    Retrieves the filename and associated ID based on the given ID and dataframe.
    
    Parameters:
    ----------
    ID : str
        The identifier, which could be either a ZTF ID or a Name present in the dataframe.
    df : pandas.DataFrame
        The dataframe containing the columns `ztf_id`, `Name` (optional), and `filename`.

    Returns:
    -------
    tuple
        A tuple containing:
        - filename (str): The corresponding filename for the given ID.
        - name (str): The ID and additional name details if available.
    """
    required_columns = {'ztf_id', 'filename'}
    if not required_columns.issubset(df.columns):
        raise KeyError(f"The dataframe must contain the following columns: {required_columns}.")
    

    # if 'Name' in df.columns:
    #     has_name_column = True
    #     ZTF_data = pd.read_csv('ZTF_TDE_catalog.csv')
    #     name = ZTF_data.loc[ZTF_data.ztf_id == ID].Name.to_list()
    #     if len(name)!=0:
    #         name_ID = name[0]
    #         has_name = True
    #     else:
    #         has_name = False
    # elif os.path.exists('ZTF_TDE_catalog.csv'):
    #     plateau_data = pd.read_csv('ZTF_TDE_catalog.csv')
    #     name = plateau_data.loc[plateau_data.ztf_id == ID].Name.to_list()
    #     has_name_column = False
    #     if len(name)!=0:
    #         name_ID = name[0]
    #         has_name = True
    #     else:
    #         has_name = False

    has_label_column = 'label' in df.columns
    

    filename = None
    name = None

    if ID in df.Name.values:
        match = df.loc[df.Name == ID]
        if not match.empty:
            filename = match.iloc[0].filename
            ztf_id = match.iloc[0].ztf_id
            name = match.iloc[0].Name
            
            if ztf_id==name:
                name = ''

            label = match.iloc[0].label
        else:
            raise ValueError(f"ID '{ID}' not found as a Name in the dataframe.")
    else:
        raise ValueError(f"ID '{ID}' not recognized in the dataframe.")

    if has_label_column:
        return filename, name, ztf_id, label
    else:
        return filename, name, ztf_id, ''

def get_full_lightcurve_data(filename, name):
    """
    Retrieves the complete lightcurve data, including additional lightcurve data from the manyTDE dataset if available.

    Parameters:
    ----------
    filename : str
        The filename of the lightcurve data to process.
    name : str
        The ZTF ID or unique identifier for the lightcurve.

    Returns:
    -------
    tuple
        A tuple containing:
        - lightcurve_df (pandas.DataFrame): The cleaned lightcurve data, including any additional data from the manyTDE dataset.
        - unique_filters (numpy.ndarray): Array of unique filters present in the lightcurve data.
        - peak_time (float): The time corresponding to the maximum flux in the `ZTF_r` filter, or a default if unavailable.

    """
    if filename != '':
        lightcurve_df = clean_data(filename, ZTF_ID=name)
    else:
        lightcurve_df = pd.DataFrame()


    try:
        manyTDE_data, mTfilters, mTfilter_freq, peak_time = get_lightcurve_data(name)
    except FileNotFoundError:
        manyTDE_data = pd.DataFrame()
        mTfilters = []
        peak_time = lightcurve_df.loc[lightcurve_df['filter'] == 'ZTF_r'].time.to_numpy()[
            np.argmax(lightcurve_df.loc[lightcurve_df['filter'] == 'ZTF_r'].flux.to_numpy())
        ]

    disabled_filters = ['r.ztf', 'g.ztf', 'i.ztf', 'W1.wise', 'W2.wise'] if filename!='' else ['W1.wise', 'W2.wise']
    for filter_name in mTfilters:
        if filter_name not in disabled_filters:
            filter_data = pd.DataFrame(
                np.array(manyTDE_data[filter_name]).T,
                columns=['time', 'flux', 'flux_unc']
            )
            filter_data[['fluxDN', 'flux_uncDN', 'zeropoint']] = 0, 0, 0
            if filter_name == 'r.ztf':
                filter_data['filter'] = 'ZTF_r'
            elif filter_name == 'g.ztf':
                filter_data['filter'] = 'ZTF_g'
            elif filter_name == 'i.ztf':
                filter_data['filter'] = 'ZTF_i'
            else:
                filter_data['filter'] = filter_name
            lightcurve_df = pd.concat((lightcurve_df, filter_data), ignore_index=True)



    unique_filters = np.unique(lightcurve_df['filter'])

    return lightcurve_df, unique_filters, peak_time


def baseline_correction_4filter(flux, flux_unc, time,  goback = 100, threshold_obs = 30, unit = 'uJy', peak_time = 0):

    '''
    Also chi square correction to 1
    '''
    flag = False
    crctn_data = {}

    len_gobackd = sum(time<(peak_time - goback))
    
    if len_gobackd>threshold_obs:
        flux_base = flux[time<(peak_time - goback)]
        flux_unc_base = flux_unc[time<(peak_time - goback)]
        avg, avg_unc = inverse_variance_weighting(flux_base, flux_unc_base)
        
        baseline_crctd_flux = flux - avg
        baseline_crctd_flux_unc = np.sqrt(flux_unc**2 + avg_unc**2)
        
        # setting pre_peak chi_square to 1
        chi_square_pre_peak = np.sum((baseline_crctd_flux[time<(peak_time - goback)] - avg)**2 / baseline_crctd_flux_unc[time<(peak_time - goback)]**2) * 1/(len(flux_base)-1)
        baseline_crctd_flux_unc * np.sqrt(chi_square_pre_peak)


        crctn_data['offset'] = avg
        crctn_data['used_points'] = len_gobackd
        flag = True
        return baseline_crctd_flux, baseline_crctd_flux_unc, crctn_data, flag
    else:
        crctn_data['used_points'] = len_gobackd
        return flux, flux_unc, crctn_data, flag

def get_flux_data4fil(data, fil, unit, baseline_correction_args ):

    # setting the units of flux for data extraction
    flux_suffix = '' if unit == "uJy" else 'DN'

    BL_burnback, obs_num_threshold, peak_time = baseline_correction_args


    flux = data.loc[data['filter'] == fil]['flux'+flux_suffix].to_numpy()
    fluxerr = data.loc[data['filter'] == fil]['flux_unc'+flux_suffix].to_numpy()
    time = data.loc[data['filter'] == fil]['time'].to_numpy()

    # Baseline correction, BLcrctn_flag =0 if no correction happened, 1 if it did
    baseline_crctd_flux, baseline_crctd_fluxerr ,baseline_crctn_metadata, BLcrctn_flag = baseline_correction_4filter(flux, fluxerr, time,  
                                                                                            goback=BL_burnback, 
                                                                                            threshold_obs=obs_num_threshold, 
                                                                                            unit=unit, 
                                                                                            peak_time =  peak_time)
        
    return time, flux, fluxerr, baseline_crctd_flux, baseline_crctd_fluxerr, baseline_crctn_metadata, BLcrctn_flag







###############################################################
#########        PLOTTING function
###############################################################



def plot_ztf_lightcurves(filename, filtr, name, axes, flux_unit = 'uJy', peak_zero = False, BL_burnback = 200, threshold_obs = 30, BL_crctn_trig = True, 
                         dbl_baseline =None, modelling = True, temp_evol = 'linear', data_opacity = 0.5, tde_model = 'SigmoidxExp', weight = 1):



 
    # Collecting the data including manyTDE dataset (if present)
    lightcurve_data, filter_avble, peak_time = get_full_lightcurve_data(filename, name)
    
    ax00 = axes[0]



    LOG_msgs = []   
    INFO = []


    filtr_present = list(set(filtr) & set(filter_avble))
    
    unavailble_filters = set(filtr) - set(filtr_present)
    INFO.append(sorted(filter_avble, reverse=True))

    time_dict, flux_dict, fluxerr_dict, baseline_crctd_flux_dict, baseline_crctd_fluxerr_dict = {}, {}, {}, {}, {} 


    all_BL_crctn_flag = True
    
    for findex, fil_val  in enumerate(filtr_present): 
        time_dict[fil_val], flux_dict[fil_val], fluxerr_dict[fil_val], baseline_crctd_flux_dict[fil_val], baseline_crctd_fluxerr_dict[fil_val], baseline_crctn_metadata, BL_crctn_flag = get_flux_data4fil(lightcurve_data, 
                                                                                                                                    fil_val,
                                                                                                                                    flux_unit,
                                                                                                                                    [BL_burnback, threshold_obs, peak_time])
        # correcting the time with peak time if triggered
        if findex == 0:
            key= fil_val
        elif len(time_dict[fil_val])>len(time_dict[key]):
            key = fil_val

        if peak_zero:
            time_dict[fil_val] = time_dict[fil_val] - peak_time
            new_peak_time = 0
        else:
            new_peak_time = peak_time

        if fil_val not in ['B.uvot', 'U.uvot', 'UVM2.uvot', 'UVW1.uvot', 'UVW2.uvot']:
            all_BL_crctn_flag = BL_crctn_flag and all_BL_crctn_flag

        # plotting depending on the the Baseline correction is enabled or not
        if BL_crctn_trig and BL_crctn_flag:
            ax00.errorbar(time_dict[fil_val], baseline_crctd_flux_dict[fil_val], baseline_crctd_fluxerr_dict[fil_val], fmt='s', color = colors[fil_val] , label = fil_val,alpha = data_opacity)

            INFO.append('- ' + fil_val+' offset : ' + str(np.round(baseline_crctn_metadata['offset'], decimals=2)))
            

        else:
            ax00.errorbar(time_dict[fil_val], flux_dict[fil_val], fluxerr_dict[fil_val], fmt='s', color = colors[fil_val], label = fil_val,alpha = data_opacity)

            if (not BL_crctn_trig):
                if findex == 0:
                    INFO.append('- ' + 'Baseline correction disabled')
            elif (not BL_crctn_flag):
                INFO.append('- ' + fil_val + ':  two few points('+ str(baseline_crctn_metadata['used_points']) + ')')
                
    



    ax00.legend(fontsize=5)
    plt.tight_layout()
    

    return [ax00] , LOG_msgs, INFO










def visualiser(merged_df, Names, manyTDE_bool = False):
    
    def axes_initiator(fig, flux_unit, peak_zero, title, log_scale_bool):

        ax00 = fig.add_subplot(111)
    

        ax00.xaxis.set_tick_params(labelbottom=True)
        ax00.tick_params(labelsize = max(6, int(canvas.get_tk_widget().winfo_width() / 190)))
        #print(canvas.get_tk_widget().winfo_width())
        if flux_unit =='uJy':
            ax00.set_ylabel(r'$Flux\;[\mu Jy]$', fontsize = max(8, int(canvas.get_tk_widget().winfo_width() / 150)))
        elif flux_unit == 'DN':
            ax00.set_ylabel(r'$Flux\;[DN]$', fontsize = max(8, int(canvas.get_tk_widget().winfo_width() / 150)))
        if peak_zero:
            ax00.set_xlabel(r'$MJD - t_{peak}$', fontsize = max(8, int(canvas.get_tk_widget().winfo_width() / 150)))

        else:
            ax00.set_xlabel(r'$MJD$', fontsize = max(2, int(canvas.get_tk_widget().winfo_width() / 150)))

        ax00.grid(alpha = 0.5)

        ax00.axhline(0, color = 'k', ls = '--', alpha =0.5)

        if log_scale_bool:
            ax00.set_yscale('symlog')

        return [ax00]
    
    global settings
    settings = {}

    global current_obj
    current_obj = None

    global loading_settings
    loading_settings = False

    

    def update_plot(event = None, reset =False):
        
        global current_obj

        if loading_settings:
            return
        
        new_obj = Name_dropdown.get()
        if new_obj != current_obj:
            # Save previous object's settings
            if current_obj:
                settings[current_obj]['filters'] = [filter_list.get(i) for i in filter_list.curselection()]
            
            # Update current object
            current_obj = new_obj
            
            # Initialize settings if new object
            if current_obj not in settings:
                initialize_settings()
            
            # Update filter list for new object
            update_filter_list(current_obj)
        




        global axes
        # Retrieve current axis limits (if available)
        if reset:
            axes = None
        elif (event and event.widget == Name_dropdown):
            axes = None


        Name = Name_dropdown.get()
        flux_unit = flux_unit_dropdown.get()
        
        fil = [filter_list.get(i) for i in filter_list.curselection()]

        zero_time = zero_time_var.get()
        baseline_crctn = BC_bool_var.get()
        log_scale_bool = log_scale_var.get()

        try:
            BCburnin = int(BCburnin_entry.get())
            BCthreshold = int(BCthreshold_entry.get())
        except ValueError:
            BCburnin = 50
            BCthreshold =10

        if manyTDE_bool:
            filename, alter_name, ztf_id, label = '', Name, '', ''
        else:
            filename, alter_name, ztf_id, label = get_filename_and_id(Name, merged_df)

        if flux_unit == "uJy":
            flux_unit = 'uJy'

        
        fig.clear()



        axes = axes_initiator(fig, flux_unit, zero_time, ztf_id, log_scale_bool)
        
        _, LOG_msgs, INFO = plot_ztf_lightcurves(filename, filtr = np.array(fil), name = alter_name, axes=axes, peak_zero=zero_time, BL_crctn_trig=baseline_crctn,
                                                                        flux_unit=flux_unit, BL_burnback=BCburnin, threshold_obs = BCthreshold,
                                                                        data_opacity = 0.2)

        canvas.draw()
        
        # Update details text
        details_text.configure(state='normal')
        details_text.delete(1.0, tk.END)

        details_text.insert(tk.END, f"Source info\n")
        details_text.insert(tk.END, f"-------------------------------------------\n")
        details_text.insert(tk.END, f"ZTF ID \t\t\t: {ztf_id}\n")
        details_text.insert(tk.END, f"Alternate name \t\t\t: {alter_name}\n")
        details_text.insert(tk.END, f"Available filters \t\t\t:")
        for i,val in enumerate(INFO[0]):
            if i == 0:
                details_text.insert(tk.END, f" {val}\n")
            else:
                details_text.insert(tk.END, f"\t\t\t  {val}\n")

        
        details_text.insert(tk.END, f"\nBaseline correction info (Burnin :{BCburnin}, obs threshold :{BCthreshold})\n")
        details_text.insert(tk.END, f"-------------------------------------------\n")
        for val in INFO[1:]:
            details_text.insert(tk.END, f"{val}\n")

        details_text.configure(state='disabled')





    text_input = {}




    # Initialize settings dictionary
    
    def initialize_settings():
        """Initialize or update settings for all objects"""
        global settings
        for obj in Names:
            if obj not in settings:
                # Get available filters for this object
                try:
                    _, filter_avble, _ = get_full_lightcurve_data('', obj)
                    filter_avble = sorted(filter_avble, reverse=True)
                    settings[obj] = {
                        'filters': filter_avble[:2],
                    }
                except UnboundLocalError:
                    pass
                #print(settings.keys())


    def update_filter_list(obj_name):
        """Update the filter listbox for the selected object"""
        global loading_settings
        loading_settings = True
        
        # Get available filters for current object
        _, filter_avble, _ = get_full_lightcurve_data('', obj_name)
        
        # Clear existing filters
        filter_list.delete(0, tk.END)
        
        # Insert new filters
        for col in sorted(filter_avble, reverse=True):
            filter_list.insert(tk.END, col)
        
        # Restore selections from settings
        filter_list.selection_clear(0, tk.END)
        saved_filters = settings[obj_name]['filters']
        for idx, col in enumerate(sorted(filter_avble,reverse=True)):
            if col in saved_filters:
                filter_list.selection_set(idx)
        
        loading_settings = False


    global root
    root = tk.Tk()
    root.title("TDEye")
    root.configure(bg="black")
    
    root.rowconfigure(0, weight=1)  # Plot frame expands
    root.rowconfigure(1, weight=0)  # Control frame fixed
    root.columnconfigure(0, weight=0)

    
    plot_frame = tk.Frame(root, bg="black")
    plot_frame.grid(row=0, column=0, sticky="ew")
    plot_frame.rowconfigure(0, weight=1)
    plot_frame.columnconfigure(0, weight=1)


    bottom_frame = tk.Frame(root, bg="black")
    bottom_frame.grid(row=1, column=0, sticky="nsew")
    bottom_frame.rowconfigure(0, weight=1)
    bottom_frame.columnconfigure(0, weight=1)
    bottom_frame.columnconfigure(1, weight=1)

    control_frame = tk.Frame(bottom_frame, bg="black", padx=10, pady=10)
    control_frame.grid(row=0, column=0, sticky="nsew")

    details_frame = tk.Frame(bottom_frame, bg="black", padx=10, pady=10)
    details_frame.grid(row=0, column=1, sticky="nsew")

    control_frame.rowconfigure(0, weight=0)
    control_frame.rowconfigure(1, weight=0)
    control_frame.rowconfigure(2, weight=0)
    control_frame.rowconfigure(3, weight=0)
    control_frame.rowconfigure(4, weight=0)
    control_frame.rowconfigure(5, weight=0)
    control_frame.rowconfigure(6, weight=0)
    control_frame.rowconfigure(7, weight=0)
    control_frame.rowconfigure(9, weight=0)
    control_frame.rowconfigure(10, weight=0)
    control_frame.rowconfigure(11, weight=0)
    control_frame.columnconfigure(0, weight=1)
    control_frame.columnconfigure(1, weight=1)
    control_frame.columnconfigure(2, weight=1)
    control_frame.columnconfigure(3, weight=1)






    # Group:OBJ name list
    tk.Label(control_frame, text="OBJ Name:", bg="black", fg="white").grid(row=0, column=0, sticky="w", pady=2)
    Name_dropdown = ttk.Combobox(control_frame, values=Names)
    Name_dropdown.grid(row=0, column=1, columnspan=4, sticky="ew", padx=5, pady=2)
    Name_dropdown.current(0)


    tk.Label(control_frame, text="Filter Selection:", bg="black", fg="white").grid(row=1, column=0, sticky="w", pady=2)
    filter_list = tk.Listbox(control_frame, selectmode=tk.MULTIPLE, height=4, exportselection=False, bg="black", fg="white")
    filter_list.grid(row=1, column=1, columnspan=4, rowspan=4, sticky="ew", padx=5, pady=2)




    customisation_frame = tk.LabelFrame(control_frame, text="Plot customisation", padx=5, pady=5, bg="black", fg="white")
    customisation_frame.grid(row=5, column=0, sticky='ew', columnspan=2, rowspan=4)
    customisation_frame.rowconfigure(0, weight=0)
    customisation_frame.rowconfigure(1, weight=0)
    customisation_frame.columnconfigure(0, weight=1)
    customisation_frame.columnconfigure(1, weight=1)
    customisation_frame.columnconfigure(2, weight=1)
    #customisation_frame.columnconfigure(4, weight=1)

    tk.Label(customisation_frame, text="Flux unit:", bg="black", fg="white").grid(row=0, column=0, sticky="w", pady=2)
    flux_unit_dropdown = ttk.Combobox(customisation_frame, values=["uJy", "DN"])
    flux_unit_dropdown.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
    flux_unit_dropdown.current(0)

    zero_time_var = tk.BooleanVar(value=True)
    tk.Checkbutton(customisation_frame, text="Zero time", variable=zero_time_var, command=update_plot,
                bg="black", fg="white", selectcolor="black").grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

    log_scale_var = tk.BooleanVar(value=False)
    tk.Checkbutton(customisation_frame, text="Log scale", variable=log_scale_var, command=update_plot,
                bg="black", fg="white", selectcolor="black").grid(row=2, column=0, columnspan=2, sticky="w", pady=2)



    # # Group: Baseline correction list
    BC_frame = tk.LabelFrame(control_frame, text="Baseline correction", padx=5, pady=5, bg="black", fg="white")
    BC_frame.grid(row=5, column=2, sticky='ew', columnspan=2, rowspan=4)
    BC_frame.rowconfigure(0, weight=0)
    BC_frame.rowconfigure(1, weight=0)
    BC_frame.rowconfigure(2, weight=0)
    BC_frame.columnconfigure(0, weight=0)
    BC_frame.columnconfigure(1, weight=0)
    

    BC_bool_var = tk.BooleanVar(value=True)
    tk.Checkbutton(BC_frame, text="Baseline correction", variable=BC_bool_var, command=update_plot,
                bg="black", fg="white", selectcolor="black").grid(row=0, column=0, columnspan=1, sticky="w", pady=2)

    
    tk.Label(BC_frame, text="burnin from peak:", bg="black", fg="white").grid(row=1, column=0, sticky="w", pady=2)
    BCburnin_var = tk.StringVar()
    BCburnin_entry = tk.Entry(
        BC_frame,
        textvariable=BCburnin_var,
        bg="black",
        fg="white",
        insertbackground="white"
    )
    BCburnin_entry.grid(row=1, column=2, columnspan=1, sticky="ew", padx=5, pady=2)
    BCburnin_var.set("50")

    tk.Label(BC_frame, text="Min obs required:", bg="black", fg="white").grid(row=2, column=0, sticky="w", pady=2)
    BCthreshold_var = tk.StringVar()
    BCthreshold_entry = tk.Entry(
        BC_frame,
        textvariable=BCthreshold_var,
        bg="black",
        fg="white",
        insertbackground="white"
    )
    BCthreshold_entry.grid(row=2, column=2, columnspan=1, sticky="ew", padx=5, pady=2)
    BCthreshold_var.set("10")



    # Configure details frame
    details_frame.rowconfigure(0, weight=1)
    details_frame.columnconfigure(0, weight=1)
    details_text = tk.Text(details_frame, bg="black", fg="white", wrap=tk.WORD)
    details_text.grid(row=0, column=0, sticky="nsew")



 
    





    fig =  plt.figure()
    
    global canvas
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    
    # Customize toolbar with ttk styling
    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
    toolbar.update()

    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Create a custom style for ttk widgets
    style = ttk.Style()
    style.configure('Toolbar.TButton', 
                    background='black', 
                    foreground='white',
                    relief='flat')

    # Apply the style to all toolbar buttons
    for button in toolbar.winfo_children():
        if isinstance(button, ttk.Button):
            button.configure(style='Toolbar.TButton')
            

    # Bind dropdown and listbox to update the plot
    Name_dropdown.bind("<<ComboboxSelected>>", update_plot)
    flux_unit_dropdown.bind("<<ComboboxSelected>>", update_plot)
    filter_list.bind("<<ListboxSelect>>", update_plot)
    BCburnin_entry.bind("<KeyRelease>", update_plot)
    BCthreshold_entry.bind("<KeyRelease>", update_plot)


    


    initialize_settings()
    current_obj = Name_dropdown.get()
    update_filter_list(current_obj)
    
    update_plot()

    root.mainloop()







if __name__ == "__main__":

    if not os.path.isdir('manyTDE/data/sources/'):
        print('Clone the manyTDE repository (https://github.com/sjoertvv/manyTDE.git) before using TDEye!')
    else:
        sources = glob.glob('manyTDE/data/sources/*')
        manyTDE_sources = sorted([val[21:-5] for val in sources])

        visualiser(pd.DataFrame(), manyTDE_sources, manyTDE_bool=True)

        