from pinglib_sep.files import get_file_list
from pinglib_sep.image_color_manifold import image_color_manifold

image_paths = get_file_list('./example_data', 'jpg')
image_color_manifold(image_paths, working_dir='working_folder')

'''
    ---------------------- Improved version for visualization ---------------------------
    function visualization(manifold_path)

        show_spectrum=true;
        text_color=[0.7,0.7,0.7];

        load(manifold_path)

        anchor_amount=size(anchor_coords,1);
        float_amount=size(float_coords,1);

        figure;
        axis equal;xlabel('dim\_1');ylabel('dim\_2')
        hold on

        %%  draw the scatters
        scatter(anchor_coords(:,1),anchor_coords(:,2),'b')
        if float_amount>0
            scatter(float_coords(:,1),float_coords(:,2),'r')
        end

        %%  write file name
        for i=1:anchor_amount
            text(anchor_coords(i,1),anchor_coords(i,2)-1,strrep(anchor_image(i,:),'_','\_'),'HorizontalAlignment','center','color',text_color)
        end
        if float_amount>0
            for i=1:float_amount
                text(float_coords(i,1),float_coords(i,2)-1,strrep(float_image(i,:),'_','\_'),'HorizontalAlignment','center','color',text_color)
            end
        end

        %%  draw spectrum
        if show_spectrum
            line_height=10;
            line_width=8;
            %%  for anchor images
            for i=1:anchor_amount
                this_x=anchor_coords(i,1);
                this_y=anchor_coords(i,2);
                this_cluster_amount=length(anchor_spectrum{i,1});
                line_cursor=0; %   当前组从哪里开始往上画
                for j=1:this_cluster_amount
                    this_group_r=anchor_spectrum{i,2}(j,1)/255;
                    this_group_g=anchor_spectrum{i,2}(j,2)/255;
                    this_group_b=anchor_spectrum{i,2}(j,3)/255;
                    this_group_length=anchor_spectrum{i,1}(j)*line_height;
                    line([this_x,this_x],[this_y+line_cursor,this_y+line_cursor+this_group_length],'linewidth',line_width,'color',[this_group_r,this_group_g,this_group_b])
                    line_cursor=line_cursor+this_group_length;
                end
            end
            if float_amount>0
                %%  for float images
                for i=1:float_amount
                    this_x=float_coords(i,1);
                    this_y=float_coords(i,2);
                    this_cluster_amount=length(float_spectrum{i,1});
                    line_cursor=0; %   当前组从哪里开始往上画
                    for j=1:this_cluster_amount
                        this_group_r=float_spectrum{i,2}(j,1)/255;
                        this_group_g=float_spectrum{i,2}(j,2)/255;
                        this_group_b=float_spectrum{i,2}(j,3)/255;
                        this_group_length=float_spectrum{i,1}(j)*line_height;
                        line([this_x,this_x],[this_y+line_cursor,this_y+line_cursor+this_group_length],'linewidth',line_width,'color',[this_group_r,this_group_g,this_group_b])
                        line_cursor=line_cursor+this_group_length;
                    end
                end
            end
        end
    end
'''
