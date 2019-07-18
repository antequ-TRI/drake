

build = false;
collect_data = false;
run_plot = false;
make_plot = true;
showiuplot = false;
showbxplot = false;
showbvplot = false;
showftplot = false;
showtsplot = false;

shownstepplot = false;
show2dnstepplot = false;
saveplot = true;
display = false;
one_plot = false;

ground_truth = true;

exec_name = 'passive_simulation';
out_folder = 'outputs/resultcsv';
runout_folder = 'outputs/runouts';
figures_folder = 'outputs/figures';
pdfs_folder = 'outputs/figurepdfs';
args = '--target_realtime_rate="0" --max_time_step="1e-3"'; % '--autodiff="true"'
integration_schemes = {'fixed_implicit_euler', 'implicit_euler', 'semi_explicit_euler', 'runge_kutta2', 'runge_kutta3', 'radau'};
v_s_opts = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6];
specific_v_s_ind = 4;
%v_s_opts = [1e-3, 1e-4];
accuracy_opts = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6];
specific_acc_ind = 4;

out_folder = 'outputs/truthcsv';
runout_folder = 'outputs/truthrunouts';
figures_folder = 'outputs/truthfigures';
pdfs_folder = 'outputs/truthfigurepdfs';
args = '--target_realtime_rate="0" --max_time_step="1e-5"';
integration_schemes = {'runge_kutta3', 'implicit_euler'};
accuracy_opts=[1e-10, 1e-12];
v_s_opts = [1e-4];

%% procedures
if(build)
%% build
setenv('DRAKE_PATH','/home/antequ/code/github/drake');
runstring = [' bazel build --config snopt //examples/box:' exec_name ];
system(['cd $DRAKE_PATH; bazel build --config snopt //examples/box:' exec_name ' ']);

end

if(collect_data)
%% run
system(['cd $DRAKE_PATH; mkdir -p ' out_folder]);
system(['cd $DRAKE_PATH; mkdir -p ' runout_folder]);
for integration_scheme_cell_index = 1:length(integration_schemes)
    integration_scheme=integration_schemes{integration_scheme_cell_index};
    for v_s = v_s_opts
        for target_accuracy = accuracy_opts
            filename = getfname(integration_scheme, v_s, target_accuracy);
            outcsv = [out_folder '/' filename];
            outfile = [runout_folder '/' filename];
            pass_args = [args  ' --box_v_s="' num2str(v_s) '" --accuracy="' num2str(target_accuracy) '" --integration_scheme="' integration_scheme '" --run_filename="' outcsv '"'];
            runstring = ['cd $DRAKE_PATH; ./bazel-bin/examples/box/' exec_name ' ' pass_args];
            disp(runstring)
            system([runstring ' 1>' outfile '.out 2>' outfile '.err &']);
        end
    end
end

end

if(run_plot)
%% read and plot
system(['cd $DRAKE_PATH; mkdir -p ' figures_folder]);
system(['cd $DRAKE_PATH; mkdir -p ' pdfs_folder]);
integration_scheme=integration_schemes{1};
v_s = v_s_opts(1);
target_accuracy = accuracy_opts(1);
X_vs_acc = repmat(v_s_opts',1,length(accuracy_opts));
Y_vs_acc = repmat(accuracy_opts,length(v_s_opts), 1);
Nsteps_values = cell(1,length(integration_schemes));

for integration_scheme_cell_index = 1:length(integration_schemes)
    integration_scheme=integration_schemes{integration_scheme_cell_index};
    Nsteps_vs_acc = zeros(length(v_s_opts), length(accuracy_opts));
    for v_s_index = 1:length(v_s_opts)
        v_s = v_s_opts(v_s_index);
        for acc_index = 1:length(accuracy_opts)
            target_accuracy = accuracy_opts(acc_index);
            filename = [getenv('DRAKE_PATH') '/' out_folder '/' getfname(integration_scheme, v_s, target_accuracy) '.csv'];
            result = csvread(filename, 1, 0);
            time = result(:,1);
            input_u = result(:,2);
            box_x = result(:,3);
            box_v = result(:,4);
            fric = result(:,5);
            ts_size = result(:,6);
            filetitle = [ strrep(integration_scheme, '_',' ') ', v_s ' num2str(v_s, '%.0e') ', acc ' num2str(target_accuracy, '%.0e')];
            nsteps = size(result);
            nsteps = nsteps(1);
            Nsteps_vs_acc(v_s_index, acc_index) = nsteps;
            if(make_plot)
                savestr = [getenv('DRAKE_PATH') '/' figures_folder '/' getfname(integration_scheme, v_s, target_accuracy) 'iu.fig'];
                savepdf = [getenv('DRAKE_PATH') '/' pdfs_folder '/' getfname(integration_scheme, v_s, target_accuracy) 'iu.pdf'];
                 makeplot(time, input_u, ['Input Force ' filetitle], 'Force (N)', showiuplot, saveplot, savestr, savepdf);

                savestr = [getenv('DRAKE_PATH') '/' figures_folder '/' getfname(integration_scheme, v_s, target_accuracy) 'bx.fig'];
                savepdf = [getenv('DRAKE_PATH') '/' pdfs_folder '/' getfname(integration_scheme, v_s, target_accuracy) 'bx.pdf'];
                makeplot(time, box_x, ['Box x ' filetitle], 'x (m)', showbxplot, saveplot, savestr, savepdf);

                savestr = [getenv('DRAKE_PATH') '/' figures_folder '/' getfname(integration_scheme, v_s, target_accuracy) 'bv.fig'];
                savepdf = [getenv('DRAKE_PATH') '/' pdfs_folder '/' getfname(integration_scheme, v_s, target_accuracy) 'bv.pdf'];
                makeplot(time, box_v, ['Box v ' filetitle], 'v (m/s)', showbvplot, saveplot, savestr, savepdf);

                savestr = [getenv('DRAKE_PATH') '/' figures_folder '/' getfname(integration_scheme, v_s, target_accuracy) 'ft.fig'];
                savepdf = [getenv('DRAKE_PATH') '/' pdfs_folder '/' getfname(integration_scheme, v_s, target_accuracy) 'ft.pdf'];
                makeplot(time, fric, ['Friction ' filetitle], 'Force (N)', showftplot, saveplot, savestr, savepdf);


                savestr = [getenv('DRAKE_PATH') '/' figures_folder '/' getfname(integration_scheme, v_s, target_accuracy) 'ts.fig'];
                savepdf = [getenv('DRAKE_PATH') '/' pdfs_folder '/' getfname(integration_scheme, v_s, target_accuracy) 'ts.pdf'];
                makeplot(time, ts_size, ['Step Size ' filetitle], 'Timestep Size (s)', showtsplot, saveplot, savestr, savepdf);
            end
        end
    end
    Nsteps_values{integration_scheme_cell_index} = Nsteps_vs_acc;

end


%% plot

for integration_scheme_cell_index = 1:length(integration_schemes)
    integration_scheme=integration_schemes{integration_scheme_cell_index};
    Nsteps_vs_acc = Nsteps_values{integration_scheme_cell_index};
    
    if(shownstepplot)
        f = figure();
        surf(X_vs_acc, Y_vs_acc, Nsteps_vs_acc);
        ax = f.CurrentAxes;
        set(ax, 'XScale', 'log');
        set(ax, 'YScale', 'log');
        set(ax, 'ZScale', 'log');
        title(['Number of steps, ' strrep(integration_scheme, '_',' ')])
        zlabel('N steps')
        xlabel('v_s')
        ylabel('acc')
    end
    
    if(show2dnstepplot)
        nsteps = Nsteps_vs_acc(specific_v_s_ind,:);
        f = figure();
        plot(accuracy_opts, nsteps);
        ax = f.CurrentAxes;
        set(ax, 'XScale', 'log');
        set(ax, 'YScale', 'log');
        title(['Number of steps, ' strrep(integration_scheme, '_',' ') ', v_s = ' num2str(v_s_opts(specific_v_s_ind))])
        ylabel('N steps')
        xlabel('acc')
        nsteps = Nsteps_vs_acc(:,specific_acc_ind)';
        f = figure();
        plot(v_s_opts, nsteps);
        ax = f.CurrentAxes;
        set(ax, 'XScale', 'log');
        set(ax, 'YScale', 'log');
        title(['Number of steps, ' strrep(integration_scheme, '_',' ') ', acc = ' num2str(accuracy_opts(specific_acc_ind))])
        ylabel('N steps')
        xlabel('v_s')
    end
end
%%
if(one_plot)
nsteps_implicit_euler = Nsteps_values{2}(specific_v_s_ind, :);
nsteps_rk3 = Nsteps_values{5}(specific_v_s_ind, :);
        f = figure();
        plot(accuracy_opts, nsteps_implicit_euler);
        ax = f.CurrentAxes;
        set(ax, 'XScale', 'log');
        hold on
        plot(accuracy_opts, nsteps_rk3);
        ax = f.CurrentAxes;
        set(ax, 'XScale', 'log');
        title(['Number of steps, v_s = ' num2str(v_s_opts(specific_v_s_ind))])
        ylabel('N steps')
        xlabel('acc')
end


%% ground truth computation for v_s = 1e-4
% use RK3 with 1e-5 timesteps, 1e-12 error desired
if(ground_truth)
    infile = '/home/antequ/code/github/drake/outputs/truthcsv/runge_kutta3_vs40_acc120.csv'
    gtresult = csvread(infile, 1, 0);
    gttime    = gtresult(:,1);
    gtinput_u = gtresult(:,2);
    gtbox_x   = gtresult(:,3);
    gtbox_v   = gtresult(:,4);
    gtfric    = gtresult(:,5);
    gtts_size = gtresult(:,6);
    gtnsteps  = size(gtresult);
    
end


end

function [f] = makeplot(time, y,  title_s, ylabel_s, show, save, savestr, savepdf)
if(show)
    f = figure(); %'visible','off'
else
    f = figure('visible', 'off');
end
plot(time, y);
title(title_s);
ylabel(ylabel_s);
xlabel('time (s)');

if(save)
   saveas(f, savestr);
   saveas(f, savepdf);
end

if(~show)
    close(f)
end
end


function [output] = getfname(integration_scheme, v_s, target_accuracy) 
output = [integration_scheme '_vs' num2fname(v_s) '_acc' num2fname(target_accuracy)];
end

function [output] = num2fname(num) 
baseten = - log10(num);
output = num2str(floor(baseten * 10));
end