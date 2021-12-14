import math

import dash
import dash_core_components as dcc
import dash_html_components as html
import jupyter_dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from . import log
from . import utils


def plot_loss(lp_data, dir_path=None, file_name=None, show=False):
    # lp_data = self.get_loss_plot_data()
    fig = plt.figure()

    train_losses_ma = utils.moving_average(lp_data.train_losses, 100)

    plt.plot(lp_data.train_counter, lp_data.train_losses, alpha=0.3, color='blue', zorder=0)
    plt.plot(lp_data.train_counter, train_losses_ma, color='blue', zorder=1)
    plt.scatter(lp_data.test_counter, lp_data.test_losses, color='red', zorder=2)
    plt.legend(['Train Loss', 'Train Loss MA', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    loss_name = 'Loss'
    # loss_name = self.lrnr.loss_fn
    plt.ylabel(loss_name)

    if file_name or dir_path:
        utils.plt_savefig(file_name, dir_path, lp_data.name)
    if show:
        plt.show()


#
#
# def plot_loss_and_log_multiple_old(train_counter_per_run, train_losses_per_run, test_counter_per_run=None,
#                                    test_losses_per_run=None, sup_title='Loss plot', dir_path=None, file_name=None,
#                                    show=False):
#     assert len(train_counter_per_run) == len(train_losses_per_run)
#     if test_counter_per_run is None:
#         test_counter_per_run = [[] for x in train_counter_per_run]
#         test_losses_per_run = [[] for x in train_counter_per_run]
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
#     fig.suptitle(sup_title)
#
#     for i in range(0, len(train_losses_per_run)):
#         train_losses = train_losses_per_run[i]
#         train_counter = train_counter_per_run[i]
#         test_losses = test_losses_per_run[i]
#         test_counter = test_counter_per_run[i]
#
#         log_train_losses = [math.log2(x) for x in train_losses]
#         log_test_losses = [math.log2(x) for x in test_losses]
#         log_train_counter = [math.log2(x + 1) for x in train_counter]
#         log_test_counter = [math.log2(x + 1) for x in test_counter]
#
#         ax1.plot(train_counter, train_losses, color='blue', zorder=0)
#         ax1.scatter(test_counter, test_losses, color='red', zorder=1)
#         ax1.legend(['Train Loss', 'Test Loss'], loc='upper right')
#         ax1.set_ylabel('Cross-entropy loss')
#
#         ax2.plot(train_counter, log_train_losses, color='blue', zorder=0)
#         ax2.scatter(test_counter, log_test_losses, color='red', zorder=1)
#         ax2.legend(['Log₂ Train Loss', 'Log₂ Test Loss'], loc='upper right')
#         ax2.set_ylabel('Log₂ (Cross-entropy loss)')
#         ax2.set_xlabel('number of training examples seen')
#
#         ax3.plot(log_train_counter, train_losses, color='blue', zorder=0)
#         ax3.scatter(log_test_counter, test_losses, color='red', zorder=1)
#         ax3.legend(['Train Loss', 'Test Loss'], loc='upper right')
#         ax3.set_ylabel('Cross-entropy loss')
#         ax3.set_xlabel('Log₂ number of training examples seen')
#
#         ax4.plot(log_train_counter, log_train_losses, color='blue', zorder=0)
#         ax4.scatter(log_test_counter, log_test_losses, color='red', zorder=1)
#         ax4.legend(['Log₂ Train Loss', 'Log₂ Test Loss'], loc='upper right')
#         ax4.set_ylabel('Log₂ (Cross-entropy loss)')
#         ax4.set_xlabel('Log₂ number of training examples seen')
#
#     if file_name or dir_path:
#         utils.plt_savefig(file_name, dir_path)
#     if show:
#         plt.show()

def plot_loss_comparative(lp_data_list, sup_title='Loss plot', show_train=True, show_test=True, dir_path=None,
                          file_name=None, show=False):
    _plot_loss_comparative(lp_data_list, sup_title, show_train, show_test, dir_path, file_name, show)
    _plot_log_loss_comparative(lp_data_list, "Log" + sup_title, show_train, show_test, dir_path, file_name, show)


def _plot_log_loss_comparative(lp_data_list, sup_title='Loss plot', show_train=True, show_test=True, dir_path=None,
                               file_name=None, show=False):
    lp_data_list = [
        log.LossPlotData(lpd.name, lpd.train_counter, [math.log2(x) for x in lpd.train_losses], lpd.test_counter,
                         [math.log2(x) for x in lpd.test_losses]) for lpd in lp_data_list]
    _plot_loss_comparative(lp_data_list, sup_title, show_train, show_test, dir_path, file_name, show)


def _plot_loss_comparative(lp_data_list, sup_title='Loss plot', show_train=True, show_test=True, dir_path=None,
                           file_name=None, show=False):
    for lp_data in lp_data_list:
        assert isinstance(lp_data, log.LossPlotData)
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(sup_title)
    legend_data = []
    for lp_data in lp_data_list:
        train_losses_ma = utils.moving_average(lp_data.train_losses, 100)
        if show_train:
            plt.plot(lp_data.train_counter, train_losses_ma, zorder=1)
            legend_data.append(f'{lp_data.name}(Train)')
        if show_test:
            plt.scatter(lp_data.test_counter, lp_data.test_losses, zorder=2)
            legend_data.append(f'{lp_data.name}(Test)')
    plt.xlabel('number of training examples seen')
    loss_name = 'Loss'
    # loss_name = self.lrnr.loss_fn
    plt.ylabel(loss_name)
    plt.legend(legend_data, loc='upper right')
    if file_name or dir_path:
        utils.plt_savefig(file_name, dir_path)
    if show:
        plt.show()


def plot_loss_and_log_multiple(lp_data_list, sup_title='Loss plot', dir_path=None, file_name=None, show=False):
    for lp_data in lp_data_list:
        assert isinstance(lp_data, log.LossPlotData)
    # assert len(train_counter_per_run) == len(train_losses_per_run)
    # print('len losses:', len(train_losses_per_run))

    # if test_counter_per_run is None:
    #     test_counter_per_run = [[] for x in train_counter_per_run]
    #     test_losses_per_run = [[] for x in train_counter_per_run]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    fig.suptitle(sup_title)
    for lp_data in lp_data_list:
        train_losses = lp_data.train_losses
        train_counter = lp_data.train_counter
        test_losses = lp_data.test_losses
        test_counter = lp_data.test_counter

        log_train_losses = [math.log2(x) for x in train_losses]
        log_test_losses = [math.log2(x) for x in test_losses]
        log_train_counter = [math.log2(x + 1) for x in train_counter]
        log_test_counter = [math.log2(x + 1) for x in test_counter]

        ax1.plot(train_counter, train_losses, color='blue', zorder=0)
        ax1.scatter(test_counter, test_losses, color='red', zorder=1)
        ax1.legend(['Train Loss', 'Test Loss'], loc='upper right')
        ax1.set_ylabel('Loss')

        ax2.plot(train_counter, log_train_losses, color='blue', zorder=0)
        ax2.scatter(test_counter, log_test_losses, color='red', zorder=1)
        ax2.legend(['Log₂ Train Loss', 'Log₂ Test Loss'], loc='upper right')
        ax2.set_ylabel('Log₂ (Loss)')
        ax2.set_xlabel('number of training examples seen')

        ax3.plot(log_train_counter, train_losses, color='blue', zorder=0)
        ax3.scatter(log_test_counter, test_losses, color='red', zorder=1)
        ax3.legend(['Train Loss', 'Test Loss'], loc='upper right')
        ax3.set_ylabel('Loss')
        ax3.set_xlabel('Log₂ number of training examples seen')

        ax4.plot(log_train_counter, log_train_losses, color='blue', zorder=0)
        ax4.scatter(log_test_counter, log_test_losses, color='red', zorder=1)
        ax4.legend(['Log₂ Train Loss', 'Log₂ Test Loss'], loc='upper right')
        ax4.set_ylabel('Log₂ (Loss)')
        ax4.set_xlabel('Log₂ number of training examples seen')

    if file_name or dir_path:
        utils.plt_savefig(file_name, dir_path)
    if show:
        plt.show()

    # self._show_plots()


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


def dataframe_combine_columns(df: pd.DataFrame, columns, combined_name=None):
    if combined_name is None:
        combined_name = "(" + ','.join(columns) + ")"
    df_new = df.copy(deep=False)
    df_new[combined_name] = ''
    for i, column in enumerate(columns):
        df[column] = df[column].map(str).astype(str)
        df_new[combined_name] = df_new[combined_name].astype(str) + df_new[column].astype(str)
        # df_new[combined_name] = getattr(df_new, combined_name) + getattr(df_new, column)
        if i < len(columns) - 1:
            df_new[combined_name] = df_new[combined_name] + "."
            # df_new[combined_name] = getattr(df_new, combined_name) + "_"
    df_new = df_new.drop(columns=columns)
    return df_new, combined_name


def dataframe_marginalize_column(df: pd.DataFrame, column, value):
    df_new = df.copy(deep=False)
    df_new = df_new[df_new[column] == value]
    df_new = df_new.drop(columns=column)
    assert isinstance(df_new, pd.DataFrame)
    return df_new

def print_duplicated(df,values):
    df=df.drop(columns=[values])
    dp = df.duplicated()
    dps = [x for x in dp if x==True]
    print("duplicated: ", dps)
    # nans = [x for x in dp['loss'] if x is NaN]

def dash_app_from_dataframe(df: pd.DataFrame, xyz_columns, slider_columns=[], selection_columns=[],
                            selection_types=None, dropdown=True, run=False, combine_selection_columns=False,
                            default_selection=None, min_val=None, max_val=None):
    if selection_types is None:
        selection_types = [dropdown for _ in selection_columns]
    x_name, y_name, z_name = xyz_columns
    
    df = df.copy(deep=False)
    print_duplicated(df,z_name)
    if combine_selection_columns:
        df, selection_column = dataframe_combine_columns(df, columns=selection_columns)
        selection_columns = [selection_column]
    print_duplicated(df,z_name)
    
    other_columns = [c for c in df.columns if c not in xyz_columns]
    assert set(slider_columns + selection_columns) == set(
        other_columns), f"slider + selection: {set(slider_columns + selection_columns)}\n other_columns: {set(other_columns)}"
    df2 = df.pivot(index=selection_columns + slider_columns + [x_name], columns=y_name, values=z_name)

    app = jupyter_dash.JupyterDash()

    figx = go.Figure()
    figx.update_layout(height=300, width=300,
                       margin=dict(l=20, r=20, t=20, b=20))

    figy = go.Figure()
    figy.update_layout(height=300, width=300,
                       margin=dict(l=20, r=20, t=20, b=20))

    components = [
        html.Label(f"Max{z_name}"),
        dcc.Input(id=f"max_{z_name}", type='number', placeholder=f"Max {z_name}", value=max_val),
        html.Label(f"Min {z_name}"),
        dcc.Input(id=f"min_{z_name}", type='number', placeholder=f"Min {z_name}", value=min_val),
    ]

    gr3d_callback_slots = []
    grx_callback_slots = []

    for i, column in enumerate(selection_columns):
        column_values = df[column].unique()
        options = [dict(label=val, value=val) for val in column_values]
        if default_selection is None or default_selection[i] is None:
            default_value = [options[0]["value"]]
        else:
            default_value = default_selection[i]

        components.append(html.Label(column))
        selection_type = selection_types[i]
        if selection_type == "dropdown":
            components.append(dcc.Dropdown(id=column, options=options, value=default_value, multi=True))
        elif selection_type == "checklist":
            components.append(dcc.Checklist(id=column, options=options, value=default_value))
        gr3d_callback_slots.append(dash.dependencies.Input(column, 'value'))
        grx_callback_slots.append(dash.dependencies.Input(column, 'value'))

    for column in slider_columns:
        column_values = df[column].unique()
        max_slider_value = int(column_values[len(column_values) - 1])
        components.append(html.Label(column))
        components.append(dcc.Slider(id=column,
                                     min=0, max=max_slider_value, value=max_slider_value,
                                     updatemode="drag",
                                     marks={
                                         0: '0',
                                         int(max_slider_value / 2): f'{int(max_slider_value / 2)}',
                                         max_slider_value: f"{max_slider_value}"
                                     }))
        gr3d_callback_slots.append(dash.dependencies.Input(column, 'value'))
        grx_callback_slots.append(dash.dependencies.Input(column, 'value'))

    app.layout = html.Div([
        html.Div(components, style=dict(columncount=2,
                                        # display='inline-block',
                                        marginBottom=20, marginLeft=20, marginRight=20)),
        html.Div(dcc.Graph(id='3d-figure'), style={'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([dcc.Graph(id='x-figure', figure=figx),
                  dcc.Graph(id='y-figure', figure=figy)],
                 style={'display': 'inline-block', 'vertical-align': 'top'})
    ], style=dict(columncount=2))

    def get_surface(name, checkbox_args, slider_args):
        df_surface = df2
        for arg in checkbox_args:
            df_surface = df_surface.loc[arg]
        for arg in slider_args:
            try:
                df_surface = df_surface.loc[arg]
            except KeyError:
                values = df_surface.index.get_level_values(0)
                arg = find_nearest(values, arg)
                df_surface = df_surface.loc[arg]
        xs, ys, zs = df_surface.index.values, df_surface.columns.values, df_surface.values.transpose()
        return go.Surface(x=xs, y=ys, z=zs, name=name)

    def get_outer_product(elements):
        nr = len(elements)
        combinations = []
        # This is a bit hacky. Produces a list combinations with tuples with a value for each element[i]
        for i in range(0, nr):
            if i == 0:
                combinations = [(x,) for x in elements[0]]
            else:
                combinations_new = []
                for arg in elements[i]:
                    combinations_new = combinations_new + [(*l, arg) for l in combinations]
                combinations = combinations_new
        return combinations

    def split_args(*args):
        selection_args, slider_args = args[0:len(selection_columns)], args[len(selection_columns):]
        specs = get_outer_product(selection_args)
        return selection_args, slider_args, specs

    if False:
        @app.callback(
            dash.dependencies.Output('x-figure', 'figure'),
            dash.dependencies.Input(f'min_{z_name}', 'value'),
            dash.dependencies.Input(f'max_{z_name}', 'value'),
            dash.dependencies.Input('3d-figure', 'hoverData'),
            *grx_callback_slots
        )
        def update_graphx(minz, maxz, hover, *args):
            figx = px.line(df, x=x_name, y=z_name)
            if hover is None:
                # return figx
                figx = px.line(df, x=y_name, y=z_name)
                # return figx
                return dash.no_update
            # print('foo')
            return figx

            point = hover["points"][0]
            x, y, z = point["x"], point["y"], point["z"]

            selection_args, _, specs = split_args(*args)
            dfpoint = df.copy(deep=False)
            condition = None
            for i, spec in enumerate(specs):
                condition_ = None
                for column_nr, value in enumerate(spec):
                    if column_nr == 0:
                        condition_ = dfpoint[selection_columns[column_nr]] == value
                    else:
                        condition_ = condition & dfpoint[selection_columns[column_nr]] == value
                if i == 0:
                    condition = condition_
                else:
                    condition = condition | condition_
            dfpoint = dfpoint.drop(dfpoint[~condition].index)
            dfpoint, combined_name = dataframe_combine_columns(dfpoint, selection_columns)

            figpoint = px.line(dfpoint, x=slider_columns[0], y=z_name, color=combined_name, range_y=[0, max_val])

            # return figpoint
            print("figx")
            return figx

    @app.callback(
        dash.dependencies.Output('3d-figure', 'figure'),
        dash.dependencies.Input(f'min_{z_name}', 'value'),
        dash.dependencies.Input(f'max_{z_name}', 'value'),
        *gr3d_callback_slots
    )
    def update_graph3d(min_loss, max_loss, *args):
        fig3d = go.Figure()
        fig3d.update_layout(uirevision='constant', height=700, width=600, margin=dict(l=20, r=20, t=20, b=20))
        _, slider_args, specs = split_args(*args)
        low, high = 2, 1
        color_scale = [(100, 50, 50), (50, 100, 50), (50, 50, 100), (75, 75, 75)]
        color_scale = [(f'({r * low},{g * low},{b * low})', f'({r * high},{g * high},{b * high})') for r, g, b in
                       color_scale]
        color_scale = [[[0, f'rgb{c1}'], [1, f'rgb{c2}']] for c1, c2 in color_scale]
        for i, selection_arg in enumerate(specs):
            name = '-'.join(selection_arg)
            surface = get_surface(name, selection_arg, slider_args)
            surface.colorscale, surface.showscale = color_scale[i], False
            # surface.opacity = 0.95
            fig3d.add_trace(surface)
        fig3d.update_layout(scene=dict(
            xaxis=dict(title=x_name), yaxis=dict(title=y_name), zaxis=dict(title=z_name)), showlegend=True)
        if min_loss is not None and max_loss is not None:
            fig3d.update_layout(scene=dict(zaxis=dict(range=[min_loss, max_loss])))
        return fig3d

    if run:
        app.run_server(mode="inline", host="localhost", port=8051)
    return app
