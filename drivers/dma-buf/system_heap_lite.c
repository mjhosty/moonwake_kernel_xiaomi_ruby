// SPDX-License-Identifier: GPL-2.0
/*
 * system_heap_lite - minimal /dev/dma_heap/system for 4.19 kernels that do
 * not carry the DMA-BUF heaps framework.
 *
 * It implements only what userspace needs from /dev/dma_heap/system:
 *   open(), then ioctl(DMA_HEAP_IOCTL_ALLOC) returning a dma-buf fd.
 * The ioctl ABI is identical to upstream's <uapi/linux/dma-heap.h>.
 *
 * Buffers are cached system pages (order-0), CPU-mappable through mmap(),
 * and shareable with other devices through the normal dma-buf attach/map
 * path (dma_map_sg() does the cache maintenance).
 *
 * Written against the dma_buf_ops / dma_buf_export() of this tree:
 *   - dma_buf_export() requires priv, map_dma_buf, unmap_dma_buf, release,
 *     map (kmap) and mmap.
 *   - export info uses struct dma_buf_export_info (DEFINE_DMA_BUF_EXPORT_INFO).
 *
 * UNTESTED. Load it manually first; do not autoload at boot until proven.
 */

#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/dma-buf.h>
#include <linux/dma-mapping.h>
#include <linux/err.h>
#include <linux/fcntl.h>
#include <linux/fs.h>
#include <linux/highmem.h>
#include <linux/kernel.h>
#include <linux/list.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#define HEAP_NAME		"system"
#define HEAP_CLASS_NAME		"dma_heap"
#define HEAP_MAX_ALLOC		(1ULL << 30)	/* 1 GiB sanity cap */

/* ABI identical to upstream struct dma_heap_allocation_data / ioctl. */
struct lite_heap_alloc_data {
	__u64 len;
	__u32 fd;
	__u32 fd_flags;
	__u64 heap_flags;
};

#define LITE_HEAP_IOC_MAGIC		'H'
#define LITE_HEAP_IOCTL_ALLOC \
	_IOWR(LITE_HEAP_IOC_MAGIC, 0x0, struct lite_heap_alloc_data)
#define LITE_HEAP_VALID_FD_FLAGS	(O_CLOEXEC | O_ACCMODE)
#define LITE_HEAP_VALID_HEAP_FLAGS	(0)

struct lite_buffer {
	struct sg_table table;
	struct page **pages;
	unsigned long npages;
	struct list_head attachments;
	struct mutex lock;
};

struct lite_attachment {
	struct sg_table table;
	struct device *dev;
	struct list_head list;
	bool mapped;
};

static int dup_sg_table(struct sg_table *src, struct sg_table *dst)
{
	struct scatterlist *s, *d;
	int i, ret;

	ret = sg_alloc_table(dst, src->orig_nents, GFP_KERNEL);
	if (ret)
		return ret;

	d = dst->sgl;
	for_each_sg(src->sgl, s, src->orig_nents, i) {
		sg_set_page(d, sg_page(s), s->length, s->offset);
		d = sg_next(d);
	}
	return 0;
}

static int lite_attach(struct dma_buf *dmabuf,
		       struct dma_buf_attachment *attachment)
{
	struct lite_buffer *buf = dmabuf->priv;
	struct lite_attachment *a;

	a = kzalloc(sizeof(*a), GFP_KERNEL);
	if (!a)
		return -ENOMEM;

	if (dup_sg_table(&buf->table, &a->table)) {
		kfree(a);
		return -ENOMEM;
	}

	a->dev = attachment->dev;
	INIT_LIST_HEAD(&a->list);
	attachment->priv = a;

	mutex_lock(&buf->lock);
	list_add(&a->list, &buf->attachments);
	mutex_unlock(&buf->lock);
	return 0;
}

static void lite_detach(struct dma_buf *dmabuf,
			struct dma_buf_attachment *attachment)
{
	struct lite_buffer *buf = dmabuf->priv;
	struct lite_attachment *a = attachment->priv;

	mutex_lock(&buf->lock);
	list_del(&a->list);
	mutex_unlock(&buf->lock);

	sg_free_table(&a->table);
	kfree(a);
}

static struct sg_table *lite_map_dma_buf(struct dma_buf_attachment *attachment,
					 enum dma_data_direction direction)
{
	struct lite_attachment *a = attachment->priv;
	struct lite_buffer *buf = attachment->dmabuf->priv;
	struct sg_table *table = &a->table;

	if (!dma_map_sg_attrs(attachment->dev, table->sgl, table->nents,
			      direction, 0))
		return ERR_PTR(-ENOMEM);

	mutex_lock(&buf->lock);
	a->mapped = true;
	mutex_unlock(&buf->lock);
	return table;
}

static void lite_unmap_dma_buf(struct dma_buf_attachment *attachment,
			       struct sg_table *table,
			       enum dma_data_direction direction)
{
	struct lite_attachment *a = attachment->priv;
	struct lite_buffer *buf = attachment->dmabuf->priv;

	mutex_lock(&buf->lock);
	a->mapped = false;
	mutex_unlock(&buf->lock);

	dma_unmap_sg_attrs(attachment->dev, table->sgl, table->nents,
			   direction, 0);
}

static int lite_begin_cpu_access(struct dma_buf *dmabuf,
				 enum dma_data_direction direction)
{
	struct lite_buffer *buf = dmabuf->priv;
	struct lite_attachment *a;

	mutex_lock(&buf->lock);
	list_for_each_entry(a, &buf->attachments, list) {
		if (!a->mapped)
			continue;
		dma_sync_sg_for_cpu(a->dev, a->table.sgl, a->table.nents,
				    direction);
	}
	mutex_unlock(&buf->lock);
	return 0;
}

static int lite_end_cpu_access(struct dma_buf *dmabuf,
			       enum dma_data_direction direction)
{
	struct lite_buffer *buf = dmabuf->priv;
	struct lite_attachment *a;

	mutex_lock(&buf->lock);
	list_for_each_entry(a, &buf->attachments, list) {
		if (!a->mapped)
			continue;
		dma_sync_sg_for_device(a->dev, a->table.sgl, a->table.nents,
				       direction);
	}
	mutex_unlock(&buf->lock);
	return 0;
}

static int lite_mmap(struct dma_buf *dmabuf, struct vm_area_struct *vma)
{
	struct lite_buffer *buf = dmabuf->priv;
	unsigned long addr = vma->vm_start;
	unsigned long i = vma->vm_pgoff;
	int ret;

	if (i >= buf->npages || vma_pages(vma) > buf->npages - i)
		return -EINVAL;

	for (; addr < vma->vm_end; addr += PAGE_SIZE, i++) {
		ret = remap_pfn_range(vma, addr, page_to_pfn(buf->pages[i]),
				      PAGE_SIZE, vma->vm_page_prot);
		if (ret)
			return ret;
	}
	return 0;
}

/* Old-style kmap callback, mandatory in this tree's dma_buf_export(). */
static void *lite_kmap(struct dma_buf *dmabuf, unsigned long page_num)
{
	struct lite_buffer *buf = dmabuf->priv;

	if (page_num >= buf->npages)
		return NULL;
	return kmap(buf->pages[page_num]);
}

static void lite_kunmap(struct dma_buf *dmabuf, unsigned long page_num,
			void *vaddr)
{
	struct lite_buffer *buf = dmabuf->priv;

	if (page_num < buf->npages)
		kunmap(buf->pages[page_num]);
}

static void lite_free_buffer(struct lite_buffer *buf)
{
	unsigned long i;

	for (i = 0; i < buf->npages; i++)
		if (buf->pages[i])
			__free_page(buf->pages[i]);
	kvfree(buf->pages);
	kfree(buf);
}

static void lite_release(struct dma_buf *dmabuf)
{
	struct lite_buffer *buf = dmabuf->priv;

	sg_free_table(&buf->table);
	lite_free_buffer(buf);
}

static const struct dma_buf_ops lite_buf_ops = {
	.attach = lite_attach,
	.detach = lite_detach,
	.map_dma_buf = lite_map_dma_buf,
	.unmap_dma_buf = lite_unmap_dma_buf,
	.begin_cpu_access = lite_begin_cpu_access,
	.end_cpu_access = lite_end_cpu_access,
	.map = lite_kmap,
	.unmap = lite_kunmap,
	.mmap = lite_mmap,
	.release = lite_release,
};

static int lite_allocate(unsigned long len, unsigned int fd_flags)
{
	DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
	struct lite_buffer *buf;
	struct dma_buf *dmabuf;
	unsigned long i;
	int ret;

	buf = kzalloc(sizeof(*buf), GFP_KERNEL);
	if (!buf)
		return -ENOMEM;

	INIT_LIST_HEAD(&buf->attachments);
	mutex_init(&buf->lock);
	buf->npages = len >> PAGE_SHIFT;

	buf->pages = kvmalloc_array(buf->npages, sizeof(*buf->pages),
				    GFP_KERNEL | __GFP_ZERO);
	if (!buf->pages) {
		kfree(buf);
		return -ENOMEM;
	}

	for (i = 0; i < buf->npages; i++) {
		buf->pages[i] = alloc_page(GFP_HIGHUSER | __GFP_ZERO |
					   __GFP_NOWARN);
		if (!buf->pages[i]) {
			ret = -ENOMEM;
			goto err_free;
		}
	}

	ret = sg_alloc_table_from_pages(&buf->table, buf->pages,
					buf->npages, 0, len, GFP_KERNEL);
	if (ret)
		goto err_free;

	exp_info.ops = &lite_buf_ops;
	exp_info.size = len;
	exp_info.flags = fd_flags;
	exp_info.priv = buf;

	dmabuf = dma_buf_export(&exp_info);
	if (IS_ERR(dmabuf)) {
		ret = PTR_ERR(dmabuf);
		goto err_table;
	}

	ret = dma_buf_fd(dmabuf, fd_flags);
	if (ret < 0)
		dma_buf_put(dmabuf);	/* release() frees the buffer */
	return ret;

err_table:
	sg_free_table(&buf->table);
err_free:
	lite_free_buffer(buf);
	return ret;
}

static long lite_heap_ioctl(struct file *file, unsigned int cmd,
			    unsigned long arg)
{
	struct lite_heap_alloc_data data;
	unsigned long len;
	int fd;

	if (cmd != LITE_HEAP_IOCTL_ALLOC)
		return -ENOTTY;

	if (copy_from_user(&data, (void __user *)arg, sizeof(data)))
		return -EFAULT;

	if (data.fd)
		return -EINVAL;
	if (data.fd_flags & ~LITE_HEAP_VALID_FD_FLAGS)
		return -EINVAL;
	if (data.heap_flags & ~LITE_HEAP_VALID_HEAP_FLAGS)
		return -EINVAL;
	if (!data.len || data.len > HEAP_MAX_ALLOC)
		return -EINVAL;

	len = PAGE_ALIGN(data.len);

	fd = lite_allocate(len, data.fd_flags);
	if (fd < 0)
		return fd;

	data.fd = fd;
	if (copy_to_user((void __user *)arg, &data, sizeof(data))) {
		/* Cannot hand the fd back; drop it. */
		ksys_close(fd);
		return -EFAULT;
	}
	return 0;
}

static int lite_heap_open(struct inode *inode, struct file *file)
{
	nonseekable_open(inode, file);
	return 0;
}

static const struct file_operations lite_heap_fops = {
	.owner = THIS_MODULE,
	.open = lite_heap_open,
	.unlocked_ioctl = lite_heap_ioctl,
#ifdef CONFIG_COMPAT
	.compat_ioctl = lite_heap_ioctl,
#endif
};

static dev_t heap_devt;
static struct cdev heap_cdev;
static struct class *heap_class;
static struct device *heap_dev;

static char *lite_devnode(struct device *dev, umode_t *mode)
{
	if (mode)
		*mode = 0666;
	/* creates /dev/dma_heap/system, the path userspace expects */
	return kasprintf(GFP_KERNEL, "dma_heap/%s", dev_name(dev));
}

static int __init lite_heap_init(void)
{
	int ret;

	ret = alloc_chrdev_region(&heap_devt, 0, 1, HEAP_CLASS_NAME);
	if (ret)
		return ret;

	cdev_init(&heap_cdev, &lite_heap_fops);
	heap_cdev.owner = THIS_MODULE;
	ret = cdev_add(&heap_cdev, heap_devt, 1);
	if (ret)
		goto err_region;

	heap_class = class_create(THIS_MODULE, HEAP_CLASS_NAME);
	if (IS_ERR(heap_class)) {
		ret = PTR_ERR(heap_class);
		goto err_cdev;
	}
	heap_class->devnode = lite_devnode;

	heap_dev = device_create(heap_class, NULL, heap_devt, NULL, HEAP_NAME);
	if (IS_ERR(heap_dev)) {
		ret = PTR_ERR(heap_dev);
		goto err_class;
	}

	pr_info("system_heap_lite: /dev/dma_heap/%s ready\n", HEAP_NAME);
	return 0;

err_class:
	class_destroy(heap_class);
err_cdev:
	cdev_del(&heap_cdev);
err_region:
	unregister_chrdev_region(heap_devt, 1);
	return ret;
}

static void __exit lite_heap_exit(void)
{
	device_destroy(heap_class, heap_devt);
	class_destroy(heap_class);
	cdev_del(&heap_cdev);
	unregister_chrdev_region(heap_devt, 1);
}

module_init(lite_heap_init);
module_exit(lite_heap_exit);

MODULE_DESCRIPTION("Minimal /dev/dma_heap/system for kernels without dma-heap");
MODULE_LICENSE("GPL");
