function semilogy_marker(x, y, marker, interval, markersize,color)

if strcmp(marker,'--')
    semilogy(x, y, marker, 'color', color, 'linewidth',1.5);
    hold on 
else
semilogy(x, y,'color',color,'linewidth',1.5,'HandleVisibility','off');	%plots the main curve
hold on
semilogy(x(1:interval:end), y(1:interval:end), marker(1),...
    'HandleVisibility', 'off','markerfacecolor','auto','markeredgecolor',color,...
    'markersize',markersize,'linewidth',1.5);	%plots the markers
semilogy(x(1), y(1), marker,'markerfacecolor','auto',...
    'markersize',markersize,'markeredgecolor',color,'color',color,'linewidth',1.5);	%plots a dummy point for legend
% hold off
end
end