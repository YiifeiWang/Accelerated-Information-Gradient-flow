function plot_marker(X, Y, e,marker, interval, markersize,color)

	lo = Y-e;
	hi = Y+e;
	hp = patch([X; X(end:-1:1); X(1)], [lo; hi(end:-1:1); lo(1)], 'r','HandleVisibility', 'off');
	set(hp, 'facecolor', color, 'edgecolor', 'none');
	alpha(hp,0.1);
	hold on
	if strcmp(marker,'--')
	    plot(X, Y, marker, 'color', color, 'linewidth',1.5);
	else
		plot(X, Y,'color',color,'linewidth',1.5,'HandleVisibility','off');	%plots the main curve
		plot(X(1:interval:end), Y(1:interval:end), marker(1),...
		    'HandleVisibility', 'off','markerfacecolor','auto','markeredgecolor',color,...
		    'markersize',markersize,'linewidth',1.5);	%plots the markers
		plot(X(1), Y(1), marker,'markerfacecolor','auto',...
		    'markersize',markersize,'markeredgecolor',color,'color',color,'linewidth',1.5);	%plots a dummy point for legend
		% hold off
	end
end