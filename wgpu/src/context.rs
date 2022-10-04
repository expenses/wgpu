use std::{any::Any, fmt::Debug, future::Future, ops::Range, pin::Pin};

use wgt::{
    AdapterInfo, BufferAddress, BufferSize, Color, CompositeAlphaMode, DownlevelCapabilities,
    DynamicOffset, Extent3d, Features, ImageDataLayout, ImageSubresourceRange, IndexFormat, Limits,
    PresentMode, ShaderStages, SurfaceConfiguration, SurfaceStatus, TextureFormat,
    TextureFormatFeatures,
};

use crate::{
    backend::BufferMappedRange, BindGroupDescriptor, BindGroupLayoutDescriptor, Buffer,
    BufferAsyncError, BufferDescriptor, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePassInner, ComputePipelineDescriptor, Context, DeviceDescriptor, Error, ErrorFilter,
    ImageCopyBuffer, ImageCopyTexture, Maintain, MapMode, PipelineLayoutDescriptor,
    QuerySetDescriptor, RenderBundleDescriptor, RenderBundleEncoderDescriptor, RenderInner,
    RenderPassDescriptor, RenderPassInner, RenderPipelineDescriptor, RequestAdapterOptions,
    RequestDeviceError, SamplerDescriptor, ShaderModuleDescriptor, ShaderModuleDescriptorSpirV,
    Texture, TextureDescriptor, TextureViewDescriptor, UncapturedErrorHandler,
};

pub struct IdSendSync {
    inner: Box<dyn Any + Send + Sync>,
}
static_assertions::assert_impl_all!(IdSendSync: Send, Sync);

impl IdSendSync {
    pub fn upcast<T: Send + Sync + 'static>(id: T) -> Self {
        Self {
            inner: Box::new(id),
        }
    }

    /// Returns a reference to the inner id value if it is `T`.
    pub fn downcast_id<T: Send + Sync + 'static>(&self) -> &T {
        // FIXME: Better error message
        self.inner
            .downcast_ref()
            .expect("IdSendSync was downcast to the wrong type")
    }

    /// Consumes the id, returning the inner id value if it is `T`.
    pub fn into_id<T: Send + Sync + 'static>(self) -> T {
        // TODO: Box::into_inner would be more clear but it is nightly still: https://github.com/rust-lang/rust/issues/80437
        *self.inner.downcast::<T>().unwrap()
    }
}

impl Debug for IdSendSync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IdSendSync")
    }
}

/// A type erased id.
pub struct Id {
    inner: Box<dyn Any>,
}

impl Debug for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Id")
    }
}

impl Id {
    pub fn upcast<T: 'static>(id: T) -> Self {
        Self {
            inner: Box::new(id),
        }
    }

    /// Returns a reference to the inner id value if it is `T`.
    pub fn downcast_id_mut<T: 'static>(&mut self) -> &mut T {
        // FIXME: Better error message
        self.inner
            .downcast_mut()
            .expect("IdSendSync was downcast to the wrong type")
    }

    /// Consumes the id, returning the inner id value if it is `T`.
    pub fn into_id<T: 'static>(self) -> T {
        // TODO: Box::into_inner would be more clear but it is nightly still: https://github.com/rust-lang/rust/issues/80437
        *self.inner.downcast::<T>().unwrap()
    }
}

/// An object safe variant of [`Context`] implemented by all types that implement [`Context`].
pub trait DynContext: Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn instance_create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> IdSendSync;
    fn instance_request_adapter(
        &self,
        options: &RequestAdapterOptions<'_>,
    ) -> Pin<Box<dyn Future<Output = Option<IdSendSync>> + Send>>;
    #[allow(clippy::type_complexity)]
    fn adapter_request_device(
        &self,
        adapter: &IdSendSync,
        desc: &DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Pin<Box<dyn Future<Output = Result<(IdSendSync, IdSendSync), RequestDeviceError>> + Send>>;

    fn instance_poll_all_devices(&self, force_wait: bool) -> bool;
    fn adapter_is_surface_supported(&self, adapter: &IdSendSync, surface: &IdSendSync) -> bool;
    fn adapter_features(&self, adapter: &IdSendSync) -> Features;
    fn adapter_limits(&self, adapter: &IdSendSync) -> Limits;
    fn adapter_downlevel_capabilities(&self, adapter: &IdSendSync) -> DownlevelCapabilities;
    fn adapter_get_info(&self, adapter: &IdSendSync) -> AdapterInfo;
    fn adapter_get_texture_format_features(
        &self,
        adapter: &IdSendSync,
        format: TextureFormat,
    ) -> TextureFormatFeatures;
    fn surface_get_supported_formats(
        &self,
        surface: &IdSendSync,
        adapter: &IdSendSync,
    ) -> Vec<TextureFormat>;
    fn surface_get_supported_present_modes(
        &self,
        surface: &IdSendSync,
        adapter: &IdSendSync,
    ) -> Vec<PresentMode>;
    fn surface_get_supported_alpha_modes(
        &self,
        surface: &IdSendSync,
        adapter: &IdSendSync,
    ) -> Vec<CompositeAlphaMode>;
    fn surface_configure(
        &self,
        surface: &IdSendSync,
        device: &IdSendSync,
        config: &SurfaceConfiguration,
    );
    fn surface_get_current_texture(
        &self,
        surface: &IdSendSync,
    ) -> (
        Option<IdSendSync>,
        SurfaceStatus,
        Box<dyn Any + Send + Sync>,
    );
    fn surface_present(&self, texture: &IdSendSync, detail: &(dyn Any + Send + Sync));
    fn surface_texture_discard(&self, texture: &IdSendSync, detail: &(dyn Any + Send + Sync));

    fn device_features(&self, device: &IdSendSync) -> Features;
    fn device_limits(&self, device: &IdSendSync) -> Limits;
    fn device_downlevel_properties(&self, device: &IdSendSync) -> DownlevelCapabilities;
    fn device_create_shader_module(
        &self,
        device: &IdSendSync,
        desc: ShaderModuleDescriptor,
        shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> IdSendSync;
    unsafe fn device_create_shader_module_spirv(
        &self,
        device: &IdSendSync,
        desc: &ShaderModuleDescriptorSpirV,
    ) -> IdSendSync;
    fn device_create_bind_group_layout(
        &self,
        device: &IdSendSync,
        desc: &BindGroupLayoutDescriptor,
    ) -> IdSendSync;
    fn device_create_bind_group(
        &self,
        device: &IdSendSync,
        desc: &BindGroupDescriptor,
    ) -> IdSendSync;
    fn device_create_pipeline_layout(
        &self,
        device: &IdSendSync,
        desc: &PipelineLayoutDescriptor,
    ) -> IdSendSync;
    fn device_create_render_pipeline(
        &self,
        device: &IdSendSync,
        desc: &RenderPipelineDescriptor,
    ) -> IdSendSync;
    fn device_create_compute_pipeline(
        &self,
        device: &IdSendSync,
        desc: &ComputePipelineDescriptor,
    ) -> IdSendSync;
    fn device_create_buffer(&self, device: &IdSendSync, desc: &BufferDescriptor) -> IdSendSync;
    fn device_create_texture(&self, device: &IdSendSync, desc: &TextureDescriptor) -> IdSendSync;
    fn device_create_sampler(&self, device: &IdSendSync, desc: &SamplerDescriptor) -> IdSendSync;
    fn device_create_query_set(&self, device: &IdSendSync, desc: &QuerySetDescriptor)
        -> IdSendSync;
    fn device_create_command_encoder(
        &self,
        device: &IdSendSync,
        desc: &CommandEncoderDescriptor,
    ) -> IdSendSync;
    fn device_create_render_bundle_encoder(
        &self,
        device: &IdSendSync,
        desc: &RenderBundleEncoderDescriptor,
    ) -> Id;
    fn device_drop(&self, device: &IdSendSync);
    fn device_poll(&self, device: &IdSendSync, maintain: Maintain) -> bool;
    fn device_on_uncaptured_error(
        &self,
        device: &IdSendSync,
        handler: Box<dyn UncapturedErrorHandler>,
    );
    fn device_push_error_scope(&self, device: &IdSendSync, filter: ErrorFilter);
    fn device_pop_error_scope(
        &self,
        device: &IdSendSync,
    ) -> Pin<Box<dyn Future<Output = Option<Error>> + Send + 'static>>;
    fn buffer_map_async(
        &self,
        buffer: &IdSendSync,
        mode: MapMode,
        range: Range<BufferAddress>,
        callback: Box<dyn FnOnce(Result<(), BufferAsyncError>) + Send + 'static>,
    );
    fn buffer_get_mapped_range(
        &self,
        buffer: &IdSendSync,
        sub_range: Range<BufferAddress>,
    ) -> BufferMappedRange;
    fn buffer_unmap(&self, buffer: &IdSendSync);
    fn texture_create_view(&self, texture: &IdSendSync, desc: &TextureViewDescriptor)
        -> IdSendSync;

    fn surface_drop(&self, surface: &IdSendSync);
    fn adapter_drop(&self, adapter: &IdSendSync);
    fn buffer_destroy(&self, buffer: &IdSendSync);
    fn buffer_drop(&self, buffer: &IdSendSync);
    fn texture_destroy(&self, buffer: &IdSendSync);
    fn texture_drop(&self, texture: &IdSendSync);
    fn texture_view_drop(&self, texture_view: &IdSendSync);
    fn sampler_drop(&self, sampler: &IdSendSync);
    fn query_set_drop(&self, query_set: &IdSendSync);
    fn bind_group_drop(&self, bind_group: &IdSendSync);
    fn bind_group_layout_drop(&self, bind_group_layout: &IdSendSync);
    fn pipeline_layout_drop(&self, pipeline_layout: &IdSendSync);
    fn shader_module_drop(&self, shader_module: &IdSendSync);
    fn command_encoder_drop(&self, command_encoder: &IdSendSync);
    fn command_buffer_drop(&self, command_buffer: &IdSendSync);
    fn render_bundle_drop(&self, render_bundle: &IdSendSync);
    fn compute_pipeline_drop(&self, pipeline: &IdSendSync);
    fn render_pipeline_drop(&self, pipeline: &IdSendSync);

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline: &IdSendSync,
        index: u32,
    ) -> IdSendSync;
    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline: &IdSendSync,
        index: u32,
    ) -> IdSendSync;

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder: &IdSendSync,
        source: &IdSendSync,
        source_offset: BufferAddress,
        destination: &IdSendSync,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    );
    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder: &IdSendSync,
        source: ImageCopyBuffer,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder: &IdSendSync,
        source: ImageCopyTexture,
        destination: ImageCopyBuffer,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder: &IdSendSync,
        source: ImageCopyTexture,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    );

    fn command_encoder_begin_compute_pass(
        &self,
        encoder: &IdSendSync,
        desc: &ComputePassDescriptor,
    ) -> Id;
    fn command_encoder_end_compute_pass(&self, encoder: &IdSendSync, pass: &mut Id);
    fn command_encoder_begin_render_pass<'a>(
        &self,
        encoder: &IdSendSync,
        desc: &RenderPassDescriptor<'a, '_>,
    ) -> Id;
    fn command_encoder_end_render_pass(&self, encoder: &IdSendSync, pass: &mut Id);
    fn command_encoder_finish(&self, encoder: IdSendSync) -> IdSendSync;

    fn command_encoder_clear_texture(
        &self,
        encoder: &IdSendSync,
        texture: &Texture,
        subresource_range: &ImageSubresourceRange,
    );
    fn command_encoder_clear_buffer(
        &self,
        encoder: &IdSendSync,
        buffer: &Buffer,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );

    fn command_encoder_insert_debug_marker(&self, encoder: &IdSendSync, label: &str);
    fn command_encoder_push_debug_group(&self, encoder: &IdSendSync, label: &str);
    fn command_encoder_pop_debug_group(&self, encoder: &IdSendSync);

    fn command_encoder_write_timestamp(
        &self,
        encoder: &IdSendSync,
        query_set: &IdSendSync,
        query_index: u32,
    );
    fn command_encoder_resolve_query_set(
        &self,
        encoder: &IdSendSync,
        query_set: &IdSendSync,
        first_query: u32,
        query_count: u32,
        destination: &IdSendSync,
        destination_offset: BufferAddress,
    );

    fn render_bundle_encoder_finish(
        &self,
        encoder: Id,
        desc: &RenderBundleDescriptor,
    ) -> IdSendSync;
    fn queue_write_buffer(
        &self,
        queue: &IdSendSync,
        buffer: &IdSendSync,
        offset: BufferAddress,
        data: &[u8],
    );
    fn queue_validate_write_buffer(
        &self,
        queue: &IdSendSync,
        buffer: &IdSendSync,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    );
    fn queue_create_staging_buffer(
        &self,
        queue: &IdSendSync,
        size: BufferSize,
    ) -> Box<dyn QueueWriteBuffer>;
    fn queue_write_staging_buffer(
        &self,
        queue: &IdSendSync,
        buffer: &IdSendSync,
        offset: BufferAddress,
        staging_buffer: &dyn QueueWriteBuffer,
    );
    fn queue_write_texture(
        &self,
        queue: &IdSendSync,
        texture: ImageCopyTexture,
        data: &[u8],
        data_layout: ImageDataLayout,
        size: Extent3d,
    );
    fn queue_submit<'a>(
        &self,
        queue: &IdSendSync,
        command_buffers: Box<dyn Iterator<Item = IdSendSync> + 'a>,
    ) -> IdSendSync;
    fn queue_get_timestamp_period(&self, queue: &IdSendSync) -> f32;
    fn queue_on_submitted_work_done(
        &self,
        queue: &IdSendSync,
        callback: Box<dyn FnOnce() + Send + 'static>,
    );

    fn device_start_capture(&self, device: &IdSendSync);
    fn device_stop_capture(&self, device: &IdSendSync);

    // TODO: ComputePassInner
    fn compute_pass_set_pipeline(&self, pass: &mut Id, pipeline: &IdSendSync);
    fn compute_pass_set_bind_group(
        &self,
        pass: &mut Id,
        index: u32,
        bind_group: &IdSendSync,
        offsets: &[DynamicOffset],
    );
    fn compute_pass_set_push_constants(&self, pass: &mut Id, offset: u32, data: &[u8]);
    fn compute_pass_insert_debug_marker(&self, pass: &mut Id, label: &str);
    fn compute_pass_push_debug_group(&self, pass: &mut Id, group_label: &str);
    fn compute_pass_pop_debug_group(&self, pass: &mut Id);
    fn compute_pass_write_timestamp(&self, pass: &mut Id, query_set: &IdSendSync, query_index: u32);
    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut Id,
        query_set: &IdSendSync,
        query_index: u32,
    );
    fn compute_pass_end_pipeline_statistics_query(&self, pass: &mut Id);
    fn compute_pass_dispatch_workgroups(&self, pass: &mut Id, x: u32, y: u32, z: u32);
    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
    );

    fn render_bundle_encoder_set_pipeline(&self, encoder: &mut Id, pipeline: &IdSendSync);
    fn render_bundle_encoder_set_bind_group(
        &self,
        encoder: &mut Id,
        index: u32,
        bind_group: &IdSendSync,
        offsets: &[DynamicOffset],
    );
    fn render_bundle_encoder_set_index_buffer(
        &self,
        encoder: &mut Id,
        buffer: &IdSendSync,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        encoder: &mut Id,
        slot: u32,
        buffer: &IdSendSync,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn render_bundle_encoder_set_push_constants(
        &self,
        encoder: &mut Id,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    );
    fn render_bundle_encoder_draw(
        &self,
        encoder: &mut Id,
        vertices: Range<u32>,
        instances: Range<u32>,
    );
    fn render_bundle_encoder_draw_indexed(
        &self,
        encoder: &mut Id,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    );
    fn render_bundle_encoder_draw_indirect(
        &self,
        encoder: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
    );
    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        encoder: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
    );
    fn render_bundle_encoder_multi_draw_indirect(
        &self,
        encoder: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count: u32,
    );
    fn render_bundle_encoder_multi_draw_indexed_indirect(
        &self,
        encoder: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count: u32,
    );
    fn render_bundle_encoder_multi_draw_indirect_count(
        &self,
        encoder: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count_buffer: &IdSendSync,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    fn render_bundle_encoder_multi_draw_indexed_indirect_count(
        &self,
        encoder: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count_buffer: &IdSendSync,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );

    fn render_pass_set_pipeline(&self, pass: &mut Id, pipeline: &IdSendSync);
    fn render_pass_set_bind_group(
        &self,
        pass: &mut Id,
        index: u32,
        bind_group: &IdSendSync,
        offsets: &[DynamicOffset],
    );
    fn render_pass_set_index_buffer(
        &self,
        pass: &mut Id,
        buffer: &IdSendSync,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn render_pass_set_vertex_buffer(
        &self,
        pass: &mut Id,
        slot: u32,
        buffer: &IdSendSync,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn render_pass_set_push_constants(
        &self,
        pass: &mut Id,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    );
    fn render_pass_draw(&self, pass: &mut Id, vertices: Range<u32>, instances: Range<u32>);
    fn render_pass_draw_indexed(
        &self,
        pass: &mut Id,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    );
    fn render_pass_draw_indirect(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
    );
    fn render_pass_draw_indexed_indirect(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
    );
    fn render_pass_multi_draw_indirect(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count: u32,
    );
    fn render_pass_multi_draw_indexed_indirect(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count: u32,
    );
    fn render_pass_multi_draw_indirect_count(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count_buffer: &IdSendSync,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    fn render_pass_multi_draw_indexed_indirect_count(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count_buffer: &IdSendSync,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    fn render_pass_set_blend_constant(&self, pass: &mut Id, color: Color);
    fn render_pass_set_scissor_rect(&self, pass: &mut Id, x: u32, y: u32, width: u32, height: u32);
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_viewport(
        &self,
        pass: &mut Id,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    );
    fn render_pass_set_stencil_reference(&self, pass: &mut Id, reference: u32);
    fn render_pass_insert_debug_marker(&self, pass: &mut Id, label: &str);
    fn render_pass_push_debug_group(&self, pass: &mut Id, group_label: &str);
    fn render_pass_pop_debug_group(&self, pass: &mut Id);
    fn render_pass_write_timestamp(&self, pass: &mut Id, query_set: &IdSendSync, query_index: u32);
    fn render_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut Id,
        query_set: &IdSendSync,
        query_index: u32,
    );
    fn render_pass_end_pipeline_statistics_query(&self, pass: &mut Id);
    fn render_pass_execute_bundles<'a>(
        &self,
        pass: &mut Id,
        render_bundles: Box<dyn Iterator<Item = &'a IdSendSync> + 'a>,
    );
}

// Blanket impl of DynContext for all types which implement Context.
impl<T> DynContext for T
where
    T: Context + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn instance_create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> IdSendSync {
        IdSendSync::upcast(Context::instance_create_surface(
            self,
            display_handle,
            window_handle,
        ))
    }

    fn instance_request_adapter(
        &self,
        options: &RequestAdapterOptions<'_>,
    ) -> Pin<Box<dyn Future<Output = Option<IdSendSync>> + Send>> {
        let future = Context::instance_request_adapter(self, options);
        Box::pin(async move { future.await.map(IdSendSync::upcast) })
    }

    fn adapter_request_device(
        &self,
        adapter: &IdSendSync,
        desc: &DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Pin<Box<dyn Future<Output = Result<(IdSendSync, IdSendSync), RequestDeviceError>> + Send>>
    {
        let adapter = adapter.downcast_id::<T::AdapterId>();
        let future = Context::adapter_request_device(self, adapter, desc, trace_dir);

        Box::pin(async move {
            let (device, queue) = future.await?;
            Ok((IdSendSync::upcast(device), IdSendSync::upcast(queue)))
        })
    }

    fn instance_poll_all_devices(&self, force_wait: bool) -> bool {
        Context::instance_poll_all_devices(self, force_wait)
    }

    fn adapter_is_surface_supported(&self, adapter: &IdSendSync, surface: &IdSendSync) -> bool {
        let adapter = adapter.downcast_id::<T::AdapterId>();
        let surface = surface.downcast_id::<T::SurfaceId>();
        Context::adapter_is_surface_supported(self, adapter, surface)
    }

    fn adapter_features(&self, adapter: &IdSendSync) -> Features {
        let adapter = adapter.downcast_id::<T::AdapterId>();
        Context::adapter_features(self, adapter)
    }

    fn adapter_limits(&self, adapter: &IdSendSync) -> Limits {
        let adapter = adapter.downcast_id::<T::AdapterId>();
        Context::adapter_limits(self, adapter)
    }

    fn adapter_downlevel_capabilities(&self, adapter: &IdSendSync) -> DownlevelCapabilities {
        let adapter = adapter.downcast_id::<T::AdapterId>();
        Context::adapter_downlevel_capabilities(self, adapter)
    }

    fn adapter_get_info(&self, adapter: &IdSendSync) -> AdapterInfo {
        let adapter = adapter.downcast_id::<T::AdapterId>();
        Context::adapter_get_info(self, adapter)
    }

    fn adapter_get_texture_format_features(
        &self,
        adapter: &IdSendSync,
        format: TextureFormat,
    ) -> TextureFormatFeatures {
        let adapter = adapter.downcast_id::<T::AdapterId>();
        Context::adapter_get_texture_format_features(self, adapter, format)
    }

    fn surface_get_supported_formats(
        &self,
        surface: &IdSendSync,
        adapter: &IdSendSync,
    ) -> Vec<TextureFormat> {
        let surface = surface.downcast_id::<T::SurfaceId>();
        let adapter = adapter.downcast_id::<T::AdapterId>();
        Context::surface_get_supported_formats(self, surface, adapter)
    }

    fn surface_get_supported_present_modes(
        &self,
        surface: &IdSendSync,
        adapter: &IdSendSync,
    ) -> Vec<PresentMode> {
        let surface = surface.downcast_id::<T::SurfaceId>();
        let adapter = adapter.downcast_id::<T::AdapterId>();
        Context::surface_get_supported_present_modes(self, surface, adapter)
    }

    fn surface_get_supported_alpha_modes(
        &self,
        surface: &IdSendSync,
        adapter: &IdSendSync,
    ) -> Vec<CompositeAlphaMode> {
        let surface = surface.downcast_id::<T::SurfaceId>();
        let adapter = adapter.downcast_id::<T::AdapterId>();
        Context::surface_get_supported_alpha_modes(self, surface, adapter)
    }

    fn surface_configure(
        &self,
        surface: &IdSendSync,
        device: &IdSendSync,
        config: &SurfaceConfiguration,
    ) {
        let surface = surface.downcast_id::<T::SurfaceId>();
        let device = device.downcast_id::<T::DeviceId>();
        Context::surface_configure(self, surface, device, config)
    }

    fn surface_get_current_texture(
        &self,
        surface: &IdSendSync,
    ) -> (
        Option<IdSendSync>,
        SurfaceStatus,
        Box<dyn Any + Send + Sync>,
    ) {
        let surface = surface.downcast_id::<T::SurfaceId>();
        let (texture, status, detail) = Context::surface_get_current_texture(self, surface);
        let detail = Box::new(detail) as Box<dyn Any + Send + Sync>;
        (texture.map(IdSendSync::upcast), status, detail)
    }

    fn surface_present(&self, texture: &IdSendSync, detail: &(dyn Any + Send + Sync)) {
        let texture = texture.downcast_id::<T::TextureId>();
        Context::surface_present(self, texture, detail.downcast_ref().unwrap())
    }

    fn surface_texture_discard(&self, texture: &IdSendSync, detail: &(dyn Any + Send + Sync)) {
        let texture = texture.downcast_id::<T::TextureId>();
        Context::surface_texture_discard(self, texture, detail.downcast_ref().unwrap())
    }

    fn device_features(&self, device: &IdSendSync) -> Features {
        let device = device.downcast_id::<T::DeviceId>();
        Context::device_features(self, device)
    }

    fn device_limits(&self, device: &IdSendSync) -> Limits {
        let device = device.downcast_id::<T::DeviceId>();
        Context::device_limits(self, device)
    }

    fn device_downlevel_properties(&self, device: &IdSendSync) -> DownlevelCapabilities {
        let device = device.downcast_id::<T::DeviceId>();
        Context::device_downlevel_properties(self, device)
    }

    fn device_create_shader_module(
        &self,
        device: &IdSendSync,
        desc: ShaderModuleDescriptor,
        shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> IdSendSync {
        let device = device.downcast_id::<T::DeviceId>();
        IdSendSync::upcast(Context::device_create_shader_module(
            self,
            device,
            desc,
            shader_bound_checks,
        ))
    }

    unsafe fn device_create_shader_module_spirv(
        &self,
        device: &IdSendSync,
        desc: &ShaderModuleDescriptorSpirV,
    ) -> IdSendSync {
        let device = device.downcast_id::<T::DeviceId>();
        IdSendSync::upcast(Context::device_create_shader_module_spirv(
            self, device, desc,
        ))
    }

    fn device_create_bind_group_layout(
        &self,
        device: &IdSendSync,
        desc: &BindGroupLayoutDescriptor,
    ) -> IdSendSync {
        let device = device.downcast_id::<T::DeviceId>();
        IdSendSync::upcast(Context::device_create_bind_group_layout(self, device, desc))
    }

    fn device_create_bind_group(
        &self,
        device: &IdSendSync,
        desc: &BindGroupDescriptor,
    ) -> IdSendSync {
        let device = device.downcast_id::<T::DeviceId>();
        IdSendSync::upcast(Context::device_create_bind_group(self, device, desc))
    }

    fn device_create_pipeline_layout(
        &self,
        device: &IdSendSync,
        desc: &PipelineLayoutDescriptor,
    ) -> IdSendSync {
        let device = device.downcast_id::<T::DeviceId>();
        IdSendSync::upcast(Context::device_create_pipeline_layout(self, device, desc))
    }

    fn device_create_render_pipeline(
        &self,
        device: &IdSendSync,
        desc: &RenderPipelineDescriptor,
    ) -> IdSendSync {
        let device = device.downcast_id::<T::DeviceId>();
        IdSendSync::upcast(Context::device_create_render_pipeline(self, device, desc))
    }

    fn device_create_compute_pipeline(
        &self,
        device: &IdSendSync,
        desc: &ComputePipelineDescriptor,
    ) -> IdSendSync {
        let device = device.downcast_id::<T::DeviceId>();
        IdSendSync::upcast(Context::device_create_compute_pipeline(self, device, desc))
    }

    fn device_create_buffer(&self, device: &IdSendSync, desc: &BufferDescriptor) -> IdSendSync {
        let device = device.downcast_id::<T::DeviceId>();
        IdSendSync::upcast(Context::device_create_buffer(self, device, desc))
    }

    fn device_create_texture(&self, device: &IdSendSync, desc: &TextureDescriptor) -> IdSendSync {
        let device = device.downcast_id::<T::DeviceId>();
        IdSendSync::upcast(Context::device_create_texture(self, device, desc))
    }

    fn device_create_sampler(&self, device: &IdSendSync, desc: &SamplerDescriptor) -> IdSendSync {
        let device = device.downcast_id::<T::DeviceId>();
        IdSendSync::upcast(Context::device_create_sampler(self, device, desc))
    }

    fn device_create_query_set(
        &self,
        device: &IdSendSync,
        desc: &QuerySetDescriptor,
    ) -> IdSendSync {
        let device = device.downcast_id::<T::DeviceId>();
        IdSendSync::upcast(Context::device_create_query_set(self, device, desc))
    }

    fn device_create_command_encoder(
        &self,
        device: &IdSendSync,
        desc: &CommandEncoderDescriptor,
    ) -> IdSendSync {
        let device = device.downcast_id::<T::DeviceId>();
        IdSendSync::upcast(Context::device_create_command_encoder(self, device, desc))
    }

    fn device_create_render_bundle_encoder(
        &self,
        device: &IdSendSync,
        desc: &RenderBundleEncoderDescriptor,
    ) -> Id {
        let device = device.downcast_id::<T::DeviceId>();
        Id::upcast(Context::device_create_render_bundle_encoder(
            self, device, desc,
        ))
    }

    fn device_drop(&self, device: &IdSendSync) {
        let device = device.downcast_id::<T::DeviceId>();
        Context::device_drop(self, device)
    }

    fn device_poll(&self, device: &IdSendSync, maintain: Maintain) -> bool {
        let device = device.downcast_id::<T::DeviceId>();
        Context::device_poll(self, device, maintain)
    }

    fn device_on_uncaptured_error(
        &self,
        device: &IdSendSync,
        handler: Box<dyn UncapturedErrorHandler>,
    ) {
        let device = device.downcast_id::<T::DeviceId>();
        Context::device_on_uncaptured_error(self, device, handler)
    }

    fn device_push_error_scope(&self, device: &IdSendSync, filter: ErrorFilter) {
        let device = device.downcast_id::<T::DeviceId>();
        Context::device_push_error_scope(self, device, filter)
    }

    fn device_pop_error_scope(
        &self,
        device: &IdSendSync,
    ) -> Pin<Box<dyn Future<Output = Option<Error>> + Send + 'static>> {
        let device = device.downcast_id::<T::DeviceId>();
        Box::pin(Context::device_pop_error_scope(self, device))
    }

    fn buffer_map_async(
        &self,
        buffer: &IdSendSync,
        mode: MapMode,
        range: Range<BufferAddress>,
        callback: Box<dyn FnOnce(Result<(), BufferAsyncError>) + Send + 'static>,
    ) {
        let buffer = buffer.downcast_id::<T::BufferId>();
        Context::buffer_map_async(self, buffer, mode, range, callback)
    }

    fn buffer_get_mapped_range(
        &self,
        buffer: &IdSendSync,
        sub_range: Range<BufferAddress>,
    ) -> BufferMappedRange {
        let buffer = buffer.downcast_id::<T::BufferId>();
        Context::buffer_get_mapped_range(self, buffer, sub_range)
    }

    fn buffer_unmap(&self, buffer: &IdSendSync) {
        let buffer = buffer.downcast_id::<T::BufferId>();
        Context::buffer_unmap(self, buffer)
    }

    fn texture_create_view(
        &self,
        texture: &IdSendSync,
        desc: &TextureViewDescriptor,
    ) -> IdSendSync {
        let texture = texture.downcast_id::<T::TextureId>();
        IdSendSync::upcast(Context::texture_create_view(self, texture, desc))
    }

    fn surface_drop(&self, surface: &IdSendSync) {
        let surface = surface.downcast_id::<T::SurfaceId>();
        Context::surface_drop(self, surface)
    }

    fn adapter_drop(&self, adapter: &IdSendSync) {
        let adapter = adapter.downcast_id::<T::AdapterId>();
        Context::adapter_drop(self, adapter)
    }

    fn buffer_destroy(&self, buffer: &IdSendSync) {
        let buffer = buffer.downcast_id::<T::BufferId>();
        Context::buffer_destroy(self, buffer)
    }

    fn buffer_drop(&self, buffer: &IdSendSync) {
        let buffer = buffer.downcast_id::<T::BufferId>();
        Context::buffer_drop(self, buffer)
    }

    fn texture_destroy(&self, texture: &IdSendSync) {
        let texture = texture.downcast_id::<T::TextureId>();
        Context::texture_destroy(self, texture)
    }

    fn texture_drop(&self, texture: &IdSendSync) {
        let texture = texture.downcast_id::<T::TextureId>();
        Context::texture_drop(self, texture)
    }

    fn texture_view_drop(&self, texture_view: &IdSendSync) {
        let texture_view = texture_view.downcast_id::<T::TextureViewId>();
        Context::texture_view_drop(self, texture_view)
    }

    fn sampler_drop(&self, sampler: &IdSendSync) {
        let sampler = sampler.downcast_id::<T::SamplerId>();
        Context::sampler_drop(self, sampler)
    }

    fn query_set_drop(&self, query_set: &IdSendSync) {
        let query_set = query_set.downcast_id::<T::QuerySetId>();
        Context::query_set_drop(self, query_set)
    }

    fn bind_group_drop(&self, bind_group: &IdSendSync) {
        let bind_group = bind_group.downcast_id::<T::BindGroupId>();
        Context::bind_group_drop(self, bind_group)
    }

    fn bind_group_layout_drop(&self, bind_group_layout: &IdSendSync) {
        let bind_group_layout = bind_group_layout.downcast_id::<T::BindGroupLayoutId>();
        Context::bind_group_layout_drop(self, bind_group_layout)
    }

    fn pipeline_layout_drop(&self, pipeline_layout: &IdSendSync) {
        let pipeline_layout = pipeline_layout.downcast_id::<T::PipelineLayoutId>();
        Context::pipeline_layout_drop(self, pipeline_layout)
    }

    fn shader_module_drop(&self, shader_module: &IdSendSync) {
        let shader_module = shader_module.downcast_id::<T::ShaderModuleId>();
        Context::shader_module_drop(self, shader_module)
    }

    fn command_encoder_drop(&self, command_encoder: &IdSendSync) {
        let command_encoder = command_encoder.downcast_id::<T::CommandEncoderId>();
        Context::command_encoder_drop(self, command_encoder)
    }

    fn command_buffer_drop(&self, command_buffer: &IdSendSync) {
        let command_buffer = command_buffer.downcast_id::<T::CommandBufferId>();
        Context::command_buffer_drop(self, command_buffer)
    }

    fn render_bundle_drop(&self, render_bundle: &IdSendSync) {
        let render_bundle = render_bundle.downcast_id::<T::RenderBundleId>();
        Context::render_bundle_drop(self, render_bundle)
    }

    fn compute_pipeline_drop(&self, pipeline: &IdSendSync) {
        let pipeline = pipeline.downcast_id::<T::ComputePipelineId>();
        Context::compute_pipeline_drop(self, pipeline)
    }

    fn render_pipeline_drop(&self, pipeline: &IdSendSync) {
        let pipeline = pipeline.downcast_id::<T::RenderPipelineId>();
        Context::render_pipeline_drop(self, pipeline)
    }

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline: &IdSendSync,
        index: u32,
    ) -> IdSendSync {
        let pipeline = pipeline.downcast_id::<T::ComputePipelineId>();
        IdSendSync::upcast(Context::compute_pipeline_get_bind_group_layout(
            self, pipeline, index,
        ))
    }

    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline: &IdSendSync,
        index: u32,
    ) -> IdSendSync {
        let pipeline = pipeline.downcast_id::<T::RenderPipelineId>();
        IdSendSync::upcast(Context::render_pipeline_get_bind_group_layout(
            self, pipeline, index,
        ))
    }

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder: &IdSendSync,
        source: &IdSendSync,
        source_offset: BufferAddress,
        destination: &IdSendSync,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    ) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        let source = source.downcast_id::<T::BufferId>();
        let destination = destination.downcast_id::<T::BufferId>();
        Context::command_encoder_copy_buffer_to_buffer(
            self,
            encoder,
            source,
            source_offset,
            destination,
            destination_offset,
            copy_size,
        )
    }

    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder: &IdSendSync,
        source: ImageCopyBuffer,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    ) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        Context::command_encoder_copy_buffer_to_texture(
            self,
            encoder,
            source,
            destination,
            copy_size,
        )
    }

    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder: &IdSendSync,
        source: ImageCopyTexture,
        destination: ImageCopyBuffer,
        copy_size: Extent3d,
    ) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        Context::command_encoder_copy_texture_to_buffer(
            self,
            encoder,
            source,
            destination,
            copy_size,
        )
    }

    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder: &IdSendSync,
        source: ImageCopyTexture,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    ) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        Context::command_encoder_copy_texture_to_texture(
            self,
            encoder,
            source,
            destination,
            copy_size,
        )
    }

    fn command_encoder_begin_compute_pass(
        &self,
        encoder: &IdSendSync,
        desc: &ComputePassDescriptor,
    ) -> Id {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        Id::upcast(Context::command_encoder_begin_compute_pass(
            self, encoder, desc,
        ))
    }

    fn command_encoder_end_compute_pass(&self, encoder: &IdSendSync, pass: &mut Id) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        let pass = pass.downcast_id_mut::<T::ComputePassId>();
        Context::command_encoder_end_compute_pass(self, encoder, pass)
    }

    fn command_encoder_begin_render_pass<'a>(
        &self,
        encoder: &IdSendSync,
        desc: &RenderPassDescriptor<'a, '_>,
    ) -> Id {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        Id::upcast(Context::command_encoder_begin_render_pass(
            self, encoder, desc,
        ))
    }

    fn command_encoder_end_render_pass(&self, encoder: &IdSendSync, pass: &mut Id) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        Context::command_encoder_end_render_pass(self, encoder, pass)
    }

    fn command_encoder_finish(&self, encoder: IdSendSync) -> IdSendSync {
        let encoder = encoder.into_id::<T::CommandEncoderId>();
        IdSendSync::upcast(Context::command_encoder_finish(self, encoder))
    }

    fn command_encoder_clear_texture(
        &self,
        encoder: &IdSendSync,
        texture: &Texture,
        subresource_range: &ImageSubresourceRange,
    ) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        Context::command_encoder_clear_texture(self, encoder, texture, subresource_range)
    }

    fn command_encoder_clear_buffer(
        &self,
        encoder: &IdSendSync,
        buffer: &Buffer,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        Context::command_encoder_clear_buffer(self, encoder, buffer, offset, size)
    }

    fn command_encoder_insert_debug_marker(&self, encoder: &IdSendSync, label: &str) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        Context::command_encoder_insert_debug_marker(self, encoder, label)
    }

    fn command_encoder_push_debug_group(&self, encoder: &IdSendSync, label: &str) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        Context::command_encoder_push_debug_group(self, encoder, label)
    }

    fn command_encoder_pop_debug_group(&self, encoder: &IdSendSync) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        Context::command_encoder_pop_debug_group(self, encoder)
    }

    fn command_encoder_write_timestamp(
        &self,
        encoder: &IdSendSync,
        query_set: &IdSendSync,
        query_index: u32,
    ) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        let query_set = query_set.downcast_id::<T::QuerySetId>();
        Context::command_encoder_write_timestamp(self, encoder, query_set, query_index)
    }

    fn command_encoder_resolve_query_set(
        &self,
        encoder: &IdSendSync,
        query_set: &IdSendSync,
        first_query: u32,
        query_count: u32,
        destination: &IdSendSync,
        destination_offset: BufferAddress,
    ) {
        let encoder = encoder.downcast_id::<T::CommandEncoderId>();
        let query_set = query_set.downcast_id::<T::QuerySetId>();
        let destination = destination.downcast_id::<T::BufferId>();
        Context::command_encoder_resolve_query_set(
            self,
            encoder,
            query_set,
            first_query,
            query_count,
            destination,
            destination_offset,
        )
    }

    fn render_bundle_encoder_finish(
        &self,
        encoder: Id,
        desc: &RenderBundleDescriptor,
    ) -> IdSendSync {
        let encoder = encoder.into_id::<T::RenderBundleEncoderId>();
        IdSendSync::upcast(Context::render_bundle_encoder_finish(self, encoder, desc))
    }

    fn queue_write_buffer(
        &self,
        queue: &IdSendSync,
        buffer: &IdSendSync,
        offset: BufferAddress,
        data: &[u8],
    ) {
        let queue = queue.downcast_id::<T::QueueId>();
        let buffer = buffer.downcast_id::<T::BufferId>();
        Context::queue_write_buffer(self, queue, buffer, offset, data)
    }

    fn queue_validate_write_buffer(
        &self,
        queue: &IdSendSync,
        buffer: &IdSendSync,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) {
        let queue = queue.downcast_id::<T::QueueId>();
        let buffer = buffer.downcast_id::<T::BufferId>();
        Context::queue_validate_write_buffer(self, queue, buffer, offset, size)
    }

    fn queue_create_staging_buffer(
        &self,
        queue: &IdSendSync,
        size: BufferSize,
    ) -> Box<dyn QueueWriteBuffer> {
        let queue = queue.downcast_id::<T::QueueId>();
        Context::queue_create_staging_buffer(self, queue, size)
    }

    fn queue_write_staging_buffer(
        &self,
        queue: &IdSendSync,
        buffer: &IdSendSync,
        offset: BufferAddress,
        staging_buffer: &dyn QueueWriteBuffer,
    ) {
        let queue = queue.downcast_id::<T::QueueId>();
        let buffer = buffer.downcast_id::<T::BufferId>();
        Context::queue_write_staging_buffer(self, queue, buffer, offset, staging_buffer)
    }

    fn queue_write_texture(
        &self,
        queue: &IdSendSync,
        texture: ImageCopyTexture,
        data: &[u8],
        data_layout: ImageDataLayout,
        size: Extent3d,
    ) {
        let queue = queue.downcast_id::<T::QueueId>();
        Context::queue_write_texture(self, queue, texture, data, data_layout, size)
    }

    fn queue_submit<'a>(
        &self,
        queue: &IdSendSync,
        command_buffers: Box<dyn Iterator<Item = IdSendSync> + 'a>,
    ) -> IdSendSync {
        let queue = queue.downcast_id::<T::QueueId>();
        let command_buffers = command_buffers.into_iter().map(IdSendSync::into_id);
        IdSendSync::upcast(Context::queue_submit(self, queue, command_buffers))
    }

    fn queue_get_timestamp_period(&self, queue: &IdSendSync) -> f32 {
        let queue = queue.downcast_id::<T::QueueId>();
        Context::queue_get_timestamp_period(self, queue)
    }

    fn queue_on_submitted_work_done(
        &self,
        queue: &IdSendSync,
        callback: Box<dyn FnOnce() + Send + 'static>,
    ) {
        let queue = queue.downcast_id::<T::QueueId>();
        Context::queue_on_submitted_work_done(self, queue, callback)
    }

    fn device_start_capture(&self, device: &IdSendSync) {
        let device = device.downcast_id::<T::DeviceId>();
        Context::device_start_capture(self, device)
    }

    fn device_stop_capture(&self, device: &IdSendSync) {
        let device = device.downcast_id::<T::DeviceId>();
        Context::device_stop_capture(self, device)
    }

    fn compute_pass_set_pipeline(&self, pass: &mut Id, pipeline: &IdSendSync) {
        let pass = pass.downcast_id_mut::<T::ComputePassId>();
        let pipeline = pipeline.downcast_id::<T::ComputePipelineId>();
        ComputePassInner::set_pipeline(pass, pipeline)
    }

    fn compute_pass_set_bind_group(
        &self,
        pass: &mut Id,
        index: u32,
        bind_group: &IdSendSync,
        offsets: &[DynamicOffset],
    ) {
        let pass = pass.downcast_id_mut::<T::ComputePassId>();
        let bind_group = bind_group.downcast_id::<T::BindGroupId>();
        ComputePassInner::set_bind_group(pass, index, bind_group, offsets)
    }

    fn compute_pass_set_push_constants(&self, pass: &mut Id, offset: u32, data: &[u8]) {
        let pass = pass.downcast_id_mut::<T::ComputePassId>();
        ComputePassInner::set_push_constants(pass, offset, data)
    }

    fn compute_pass_insert_debug_marker(&self, pass: &mut Id, label: &str) {
        let pass = pass.downcast_id_mut::<T::ComputePassId>();
        ComputePassInner::insert_debug_marker(pass, label)
    }

    fn compute_pass_push_debug_group(&self, pass: &mut Id, group_label: &str) {
        let pass = pass.downcast_id_mut::<T::ComputePassId>();
        ComputePassInner::push_debug_group(pass, group_label)
    }

    fn compute_pass_pop_debug_group(&self, pass: &mut Id) {
        let pass = pass.downcast_id_mut::<T::ComputePassId>();
        ComputePassInner::pop_debug_group(pass)
    }

    fn compute_pass_write_timestamp(
        &self,
        pass: &mut Id,
        query_set: &IdSendSync,
        query_index: u32,
    ) {
        let pass = pass.downcast_id_mut::<T::ComputePassId>();
        let query_set = query_set.downcast_id::<T::QuerySetId>();
        ComputePassInner::write_timestamp(pass, query_set, query_index)
    }

    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut Id,
        query_set: &IdSendSync,
        query_index: u32,
    ) {
        let pass = pass.downcast_id_mut::<T::ComputePassId>();
        let query_set = query_set.downcast_id::<T::QuerySetId>();
        ComputePassInner::begin_pipeline_statistics_query(pass, query_set, query_index)
    }

    fn compute_pass_end_pipeline_statistics_query(&self, pass: &mut Id) {
        let pass = pass.downcast_id_mut::<T::ComputePassId>();
        ComputePassInner::end_pipeline_statistics_query(pass)
    }

    fn compute_pass_dispatch_workgroups(&self, pass: &mut Id, x: u32, y: u32, z: u32) {
        let pass = pass.downcast_id_mut::<T::ComputePassId>();
        ComputePassInner::dispatch_workgroups(pass, x, y, z)
    }

    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
    ) {
        let pass = pass.downcast_id_mut::<T::ComputePassId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        ComputePassInner::dispatch_workgroups_indirect(pass, indirect_buffer, indirect_offset)
    }

    fn render_bundle_encoder_set_pipeline(&self, encoder: &mut Id, pipeline: &IdSendSync) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        let pipeline = pipeline.downcast_id::<T::RenderPipelineId>();
        RenderInner::set_pipeline(encoder, pipeline)
    }

    fn render_bundle_encoder_set_bind_group(
        &self,
        encoder: &mut Id,
        index: u32,
        bind_group: &IdSendSync,
        offsets: &[DynamicOffset],
    ) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        let bind_group = bind_group.downcast_id::<T::BindGroupId>();
        RenderInner::set_bind_group(encoder, index, bind_group, offsets)
    }

    fn render_bundle_encoder_set_index_buffer(
        &self,
        encoder: &mut Id,
        buffer: &IdSendSync,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        let buffer = buffer.downcast_id::<T::BufferId>();
        RenderInner::set_index_buffer(encoder, buffer, index_format, offset, size)
    }

    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        encoder: &mut Id,
        slot: u32,
        buffer: &IdSendSync,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        let buffer = buffer.downcast_id::<T::BufferId>();
        RenderInner::set_vertex_buffer(encoder, slot, buffer, offset, size)
    }

    fn render_bundle_encoder_set_push_constants(
        &self,
        encoder: &mut Id,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    ) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        RenderInner::set_push_constants(encoder, stages, offset, data)
    }

    fn render_bundle_encoder_draw(
        &self,
        encoder: &mut Id,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        RenderInner::draw(encoder, vertices, instances)
    }

    fn render_bundle_encoder_draw_indexed(
        &self,
        encoder: &mut Id,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        RenderInner::draw_indexed(encoder, indices, base_vertex, instances)
    }

    fn render_bundle_encoder_draw_indirect(
        &self,
        encoder: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
    ) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        RenderInner::draw_indirect(encoder, indirect_buffer, indirect_offset)
    }

    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        encoder: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
    ) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        RenderInner::draw_indexed_indirect(encoder, indirect_buffer, indirect_offset)
    }

    fn render_bundle_encoder_multi_draw_indirect(
        &self,
        encoder: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        RenderInner::multi_draw_indirect(encoder, indirect_buffer, indirect_offset, count)
    }

    fn render_bundle_encoder_multi_draw_indexed_indirect(
        &self,
        encoder: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        RenderInner::multi_draw_indexed_indirect(encoder, indirect_buffer, indirect_offset, count)
    }

    fn render_bundle_encoder_multi_draw_indirect_count(
        &self,
        encoder: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count_buffer: &IdSendSync,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        let count_buffer = count_buffer.downcast_id::<T::BufferId>();
        RenderInner::multi_draw_indirect_count(
            encoder,
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_buffer_offset,
            max_count,
        )
    }

    fn render_bundle_encoder_multi_draw_indexed_indirect_count(
        &self,
        encoder: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count_buffer: &IdSendSync,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) {
        let encoder = encoder.downcast_id_mut::<T::RenderBundleEncoderId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        let count_buffer = count_buffer.downcast_id::<T::BufferId>();
        RenderInner::multi_draw_indexed_indirect_count(
            encoder,
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_buffer_offset,
            max_count,
        )
    }

    fn render_pass_set_pipeline(&self, pass: &mut Id, pipeline: &IdSendSync) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let pipeline = pipeline.downcast_id::<T::RenderPipelineId>();
        RenderInner::set_pipeline(pass, pipeline)
    }

    fn render_pass_set_bind_group(
        &self,
        pass: &mut Id,
        index: u32,
        bind_group: &IdSendSync,
        offsets: &[DynamicOffset],
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let bind_group = bind_group.downcast_id::<T::BindGroupId>();
        RenderInner::set_bind_group(pass, index, bind_group, offsets)
    }

    fn render_pass_set_index_buffer(
        &self,
        pass: &mut Id,
        buffer: &IdSendSync,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let buffer = buffer.downcast_id::<T::BufferId>();
        RenderInner::set_index_buffer(pass, buffer, index_format, offset, size)
    }

    fn render_pass_set_vertex_buffer(
        &self,
        pass: &mut Id,
        slot: u32,
        buffer: &IdSendSync,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let buffer = buffer.downcast_id::<T::BufferId>();
        RenderInner::set_vertex_buffer(pass, slot, buffer, offset, size)
    }

    fn render_pass_set_push_constants(
        &self,
        pass: &mut Id,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        RenderInner::set_push_constants(pass, stages, offset, data)
    }

    fn render_pass_draw(&self, pass: &mut Id, vertices: Range<u32>, instances: Range<u32>) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        RenderInner::draw(pass, vertices, instances)
    }

    fn render_pass_draw_indexed(
        &self,
        pass: &mut Id,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        RenderInner::draw_indexed(pass, indices, base_vertex, instances)
    }

    fn render_pass_draw_indirect(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        RenderInner::draw_indirect(pass, indirect_buffer, indirect_offset)
    }

    fn render_pass_draw_indexed_indirect(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        RenderInner::draw_indexed_indirect(pass, indirect_buffer, indirect_offset)
    }

    fn render_pass_multi_draw_indirect(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        RenderInner::multi_draw_indirect(pass, indirect_buffer, indirect_offset, count)
    }

    fn render_pass_multi_draw_indexed_indirect(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        RenderInner::multi_draw_indexed_indirect(pass, indirect_buffer, indirect_offset, count)
    }

    fn render_pass_multi_draw_indirect_count(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count_buffer: &IdSendSync,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        let count_buffer = count_buffer.downcast_id::<T::BufferId>();
        RenderInner::multi_draw_indirect_count(
            pass,
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_buffer_offset,
            max_count,
        )
    }

    fn render_pass_multi_draw_indexed_indirect_count(
        &self,
        pass: &mut Id,
        indirect_buffer: &IdSendSync,
        indirect_offset: BufferAddress,
        count_buffer: &IdSendSync,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let indirect_buffer = indirect_buffer.downcast_id::<T::BufferId>();
        let count_buffer = count_buffer.downcast_id::<T::BufferId>();
        RenderInner::multi_draw_indexed_indirect_count(
            pass,
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_buffer_offset,
            max_count,
        )
    }

    fn render_pass_set_blend_constant(&self, pass: &mut Id, color: Color) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        RenderPassInner::set_blend_constant(pass, color)
    }

    fn render_pass_set_scissor_rect(&self, pass: &mut Id, x: u32, y: u32, width: u32, height: u32) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        RenderPassInner::set_scissor_rect(pass, x, y, width, height)
    }

    fn render_pass_set_viewport(
        &self,
        pass: &mut Id,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        RenderPassInner::set_viewport(pass, x, y, width, height, min_depth, max_depth)
    }

    fn render_pass_set_stencil_reference(&self, pass: &mut Id, reference: u32) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        RenderPassInner::set_stencil_reference(pass, reference)
    }

    fn render_pass_insert_debug_marker(&self, pass: &mut Id, label: &str) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        RenderPassInner::insert_debug_marker(pass, label)
    }

    fn render_pass_push_debug_group(&self, pass: &mut Id, group_label: &str) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        RenderPassInner::push_debug_group(pass, group_label)
    }

    fn render_pass_pop_debug_group(&self, pass: &mut Id) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        RenderPassInner::pop_debug_group(pass)
    }

    fn render_pass_write_timestamp(&self, pass: &mut Id, query_set: &IdSendSync, query_index: u32) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let query_set = query_set.downcast_id::<T::QuerySetId>();
        RenderPassInner::write_timestamp(pass, query_set, query_index)
    }

    fn render_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut Id,
        query_set: &IdSendSync,
        query_index: u32,
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let query_set = query_set.downcast_id::<T::QuerySetId>();
        RenderPassInner::begin_pipeline_statistics_query(pass, query_set, query_index)
    }

    fn render_pass_end_pipeline_statistics_query(&self, pass: &mut Id) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        RenderPassInner::end_pipeline_statistics_query(pass)
    }

    fn render_pass_execute_bundles<'a>(
        &self,
        pass: &mut Id,
        render_bundles: Box<dyn Iterator<Item = &'a IdSendSync> + 'a>,
    ) {
        let pass = pass.downcast_id_mut::<T::RenderPassId>();
        let render_bundles = render_bundles.into_iter().map(IdSendSync::downcast_id);
        RenderPassInner::execute_bundles(pass, render_bundles)
    }
}

pub trait QueueWriteBuffer: Send + Sync {
    fn slice(&self) -> &[u8];

    fn slice_mut(&mut self) -> &mut [u8];

    fn as_any(&self) -> &dyn Any;
}

#[cfg(test)]
mod tests {
    use super::DynContext;

    fn compiles<T>() {}

    /// Assert that DynContext is object safe.
    #[test]
    fn object_safe() {
        compiles::<Box<dyn DynContext>>();
    }
}
